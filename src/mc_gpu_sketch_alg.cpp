#include "mc_gpu_sketch_alg.h"

#include <omp.h>

#include <iostream>
#include <thread>
#include <vector>

// call this function after we have found the depth of each update
void MCGPUSketchAlg::complete_update_batch(int thr_id, const TaggedUpdateBatch &updates) {
  node_id_t min_subgraph = updates.min_subgraph;
  node_id_t first_es_subgraph = updates.first_es_subgraph;

  if (first_es_subgraph == 0) {
    throw std::runtime_error("ERROR: MCGPUSketchAlg::complete_update_batch() bad es_subgraph");
  }

  // do we need to allocate more sketches due to edge_store contraction
  if (first_es_subgraph > cur_subgraphs) {
    sketch_creation_lock.lock();

    // double check to ensure no one else performed the allocation 
    if (first_es_subgraph > cur_subgraphs) {
      subgraphs[cur_subgraphs].initialize(this, cur_subgraphs, num_nodes, num_host_threads,
                                          num_device_threads, num_batch_per_buffer, batch_size,
                                          default_skt_params);
      create_sketch_graph(cur_subgraphs, subgraphs[cur_subgraphs].get_skt_params());
      cur_subgraphs++; // do this last so that threads only touch params/sketches when initialized
    }

    sketch_creation_lock.unlock();
  }

  node_id_t src_vertex = updates.src;
  auto &dsts_data = updates.dsts_data;

  constexpr size_t local_buffer_size = 32;
  std::vector<std::array<node_id_t, local_buffer_size>> subgraph_buffers;
  subgraph_buffers.resize(first_es_subgraph);
  std::array<size_t, local_buffer_size> buffer_sizes;
  buffer_sizes.fill(0);

  // put data into local buffers and when full move into subgraph's gutters
  for (size_t i = 0; i < dsts_data.size(); i++) {
    auto &dst_data = dsts_data[i];
    node_id_t update_subgraphs = std::min(dst_data.subgraph, first_es_subgraph - 1);

    for (size_t graph_id = min_subgraph; graph_id <= update_subgraphs; graph_id++) {
      subgraph_buffers[graph_id][buffer_sizes[graph_id]++] = dst_data.dst;
      if (buffer_sizes[graph_id] >= local_buffer_size) {
        subgraphs[graph_id].batch_insert(thr_id, src_vertex, subgraph_buffers[graph_id],
                                         buffer_sizes[graph_id]);
        buffer_sizes[graph_id] = 0;
      }
    }
  }

  // flush our buffers
  for (size_t i = 0; i < first_es_subgraph; i++) {
    if (buffer_sizes[i] > 0) {
      subgraphs[i].batch_insert(thr_id, src_vertex, subgraph_buffers[i], buffer_sizes[i]);
    }
  }
}

void MCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (MCSketchAlg::get_update_locked()) throw UpdateLockedException();

  node_id_t first_es_subgraph = edge_store.get_first_store_subgraph();

  // We only have an adjacency list so just directly insert
  if (first_es_subgraph == 0) {
    TaggedUpdateBatch more_upds = edge_store.insert_adj_edges(src_vertex, dst_vertices);
    if (more_upds.dsts_data.size() > 0) complete_update_batch(thr_id, more_upds);
    return;
  }

  SubgraphTaggedUpdate* store_edges = store_buffers[thr_id];
  SubgraphTaggedUpdate* sketch_edges = sketch_buffers[thr_id];

  int store_edge_count = 0;
  int sketch_edge_count = 0;
  for (vec_t i = 0; i < dst_vertices.size(); i++) {
    // Determine the depth of current edge
    vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[i]));
    size_t subgraph = Bucket_Boruvka::get_index_depth(edge_id, default_skt_params.seed, num_subgraphs-1);

    if (subgraph >= first_es_subgraph) {
      // Adj. list
      store_edges[store_edge_count] = {subgraph, dst_vertices[i]};
      store_edge_count++;
    }
    sketch_edges[sketch_edge_count] = {subgraph, dst_vertices[i]};
    sketch_edge_count++;
  }

  // Perform adjacency list updates
  TaggedUpdateBatch more_upds =
      edge_store.insert_adj_edges(src_vertex, first_es_subgraph, store_edges, store_edge_count);
  if (sketch_edge_count > 0)
    complete_update_batch(thr_id, {src_vertex, 0, first_es_subgraph, std::vector<SubgraphTaggedUpdate>(sketch_edges, sketch_edges + sketch_edge_count)});
  if (more_upds.dsts_data.size() > 0)
    complete_update_batch(thr_id, more_upds);
}

void MCGPUSketchAlg::apply_flush_updates() {
  // first ensure that all pending contractions are moved out of the edge store.
  auto task = [&](int thr_id) {
    while (edge_store.contract_in_progress()) {
      TaggedUpdateBatch more_upds =
          edge_store.vertex_advance_subgraph(edge_store.get_first_store_subgraph());
      if (more_upds.dsts_data.size() > 0) complete_update_batch(thr_id, more_upds);
    }
  };

  std::vector<std::thread> threads;
  for (size_t t = 0; t < num_host_threads; t++) {
    threads.emplace_back(task, t);
  }
  for (size_t t = 0; t < num_host_threads; t++) {
    threads[t].join();
  }

  // flush all subgraph buffers
#pragma omp parallel for
  for (size_t graph_id = 0; graph_id < cur_subgraphs; graph_id++) {
    subgraphs[graph_id].flush(get_omp_thread_num());
  }

  // ensure streams have finished applying updates
  cudaDeviceSynchronize();
}

std::vector<Edge> MCGPUSketchAlg::get_adjlist_spanning_forests() {
  return edge_store.get_edges();
}
