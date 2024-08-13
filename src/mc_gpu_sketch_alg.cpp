#include "mc_gpu_sketch_alg.h"

#include <iostream>
#include <thread>
#include <vector>

size_t MCGPUSketchAlg::get_and_apply_finished_stream(int thr_id) {
  int stream_id = thr_id * stream_multiplier;
  size_t stream_offset = 0;
  while(true) {
    int cur_stream = stream_id + stream_offset;
    if (cudaStreamQuery(streams[cur_stream].stream) == cudaSuccess) {

      // CUDA Stream is available, check if it has any delta sketch
      if(streams[cur_stream].delta_applied == 0) {

        for (int graph_id = 0; graph_id < streams[cur_stream].num_graphs; graph_id++) {   
          size_t bucket_offset = thr_id * num_buckets;
          for (size_t i = 0; i < num_buckets; i++) {
            delta_buckets[bucket_offset + i].alpha = subgraphs[graph_id].cudaUpdateParams->h_bucket_a[(cur_stream * num_buckets) + i];
            delta_buckets[bucket_offset + i].gamma = subgraphs[graph_id].cudaUpdateParams->h_bucket_c[(cur_stream * num_buckets) + i];
          }

          int prev_src = streams[cur_stream].src_vertex;
          
          if(prev_src == -1) {
            std::cout << "Stream #" << cur_stream << ": Shouldn't be here!\n";
          }

          // Apply the delta sketch
          apply_raw_buckets_update((graph_id * num_nodes) + prev_src, &delta_buckets[bucket_offset]);
        }
        streams[cur_stream].delta_applied = 1;
        streams[cur_stream].src_vertex = -1;
        streams[cur_stream].num_graphs = -1;
      }
      else {
        if (streams[cur_stream].src_vertex != -1) {
          std::cout << "Stream #" << cur_stream << ": not applying but has delta sketch: " << streams[cur_stream].src_vertex << " deltaApplied: " << streams[cur_stream].delta_applied << "\n";
        }
      }
      break;
    }
    stream_offset++;
    if (stream_offset == stream_multiplier) {
        stream_offset = 0;
    }
  }
  return stream_id + stream_offset;
}

// call this function after we have found the depth of each update
// TODO: This function may need to divide the updates into multiple update batches.
//       This is the case when a single vertex in edge store has O(n) updates.
//       Could enforce that this is the caller's responsibility.
void MCGPUSketchAlg::complete_update_batch(int thr_id, const TaggedUpdateBatch &updates) {
  int stream_id = get_and_apply_finished_stream(thr_id);
  int start_index = stream_id * batch_size;
  node_id_t min_subgraph = updates.min_subgraph;
  node_id_t first_es_subgraph = updates.first_es_subgraph;

  if (first_es_subgraph == 0) {
    std::cerr << "Why are we here??" << std::endl;
    exit(EXIT_FAILURE);
  }

  // do we need to allocate more sketches due to edge_store contraction
  if (first_es_subgraph > cur_subgraphs) {
    sketch_creation_lock.lock();

    // double check to ensure no one else performed the allocation 
    if (first_es_subgraph > cur_subgraphs) {
      create_sketch_graph(cur_subgraphs);

      CudaUpdateParams* params;
      // TODO: Is this malloc necessary?
      gpuErrchk(cudaMallocManaged(&params, sizeof(CudaUpdateParams)));
      params = new CudaUpdateParams(
         num_nodes, num_samples, num_buckets, num_columns, bkt_per_col, num_host_threads,
         num_reader_threads, batch_size, stream_multiplier, num_device_blocks, k);
      subgraphs.push_back({0, params});
      cur_subgraphs++; // do this last so that threads only touch params/sketches when initialized
    }

    sketch_creation_lock.unlock();
  }

  node_id_t src_vertex = updates.src;
  auto &dsts_data = updates.dsts_data;
  std::vector<size_t> sketch_update_size(max_sketch_graphs);
  node_id_t max_subgraph = 0;
  for (auto dst_data : dsts_data) {
    node_id_t update_subgraphs = std::min(dst_data.subgraph, first_es_subgraph - 1);
    max_subgraph = std::max(update_subgraphs, max_subgraph);
    vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_data.dst));

    for (size_t graph_id = min_subgraph; graph_id <= update_subgraphs; graph_id++) {
      subgraphs[graph_id].cudaUpdateParams->h_edgeUpdates[start_index + sketch_update_size[graph_id]] = edge_id;
      sketch_update_size[graph_id]++;
    }
  }

  // Go to every subgraph and apply sketch updates
  streams[stream_id].src_vertex = src_vertex;
  streams[stream_id].delta_applied = 0;
  streams[stream_id].num_graphs = max_subgraph + 1;

  for (int graph_id = min_subgraph; graph_id <= max_subgraph; graph_id++) {
    subgraphs[graph_id].num_updates += sketch_update_size[graph_id];
    CudaUpdateParams* cudaUpdateParams = subgraphs[graph_id].cudaUpdateParams;
    cudaMemcpyAsync(&cudaUpdateParams->d_edgeUpdates[start_index], &cudaUpdateParams->h_edgeUpdates[start_index], sketch_update_size[graph_id] * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
    cudaKernel.k_sketchUpdate(num_device_threads, num_device_blocks, streams[stream_id].stream, cudaUpdateParams->d_edgeUpdates, start_index, sketch_update_size[graph_id], stream_id * num_buckets, cudaUpdateParams, cudaUpdateParams->d_bucket_a, cudaUpdateParams->d_bucket_c, sketchSeed);
    cudaMemcpyAsync(&cudaUpdateParams->h_bucket_a[stream_id * num_buckets], &cudaUpdateParams->d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
    cudaMemcpyAsync(&cudaUpdateParams->h_bucket_c[stream_id * num_buckets], &cudaUpdateParams->d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
  }
}

void MCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (MCSketchAlg::get_update_locked()) throw UpdateLockedException();

  num_updates_seen += dst_vertices.size();

  node_id_t first_es_subgraph = edge_store.get_first_store_subgraph();

  // We only have an adjacency list so just directly insert
  if (first_es_subgraph == 0) {
    TaggedUpdateBatch more_upds = edge_store.insert_adj_edges(src_vertex, dst_vertices);
    if (more_upds.dsts_data.size() > 0) complete_update_batch(thr_id, more_upds);
    return;
  }

  // TODO: This memory allocation is sad
  std::vector<SubgraphTaggedUpdate> store_edges;
  std::vector<SubgraphTaggedUpdate> sketch_edges;

  for (vec_t i = 0; i < dst_vertices.size(); i++) {
    // Determine the depth of current edge
    vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[i]));
    size_t subgraph = Bucket_Boruvka::get_index_depth(edge_id, 0, num_subgraphs-1);

    if (subgraph >= first_es_subgraph) {
      // Adj. list
      store_edges.push_back({subgraph, dst_vertices[i]});
    }
    sketch_edges.push_back({subgraph, dst_vertices[i]});
  }

  // Perform adjacency list updates
  TaggedUpdateBatch more_upds = edge_store.insert_adj_edges(src_vertex, store_edges);
  if (sketch_edges.size() > 0)
    complete_update_batch(thr_id, {src_vertex, 0, first_es_subgraph, sketch_edges});
  if (more_upds.dsts_data.size() > 0) complete_update_batch(thr_id, more_upds);
}

void MCGPUSketchAlg::apply_flush_updates() {
  std::cerr << "Number of updates seen = " << num_updates_seen << std::endl;

  // first ensure that all pending contractions are moved out of the edge store.
  while (edge_store.contract_in_progress()) {
    TaggedUpdateBatch more_upds =
        edge_store.vertex_advance_subgraph(edge_store.get_first_store_subgraph());

    if (more_upds.dsts_data.size() > 0) complete_update_batch(0, more_upds);
  }

  // ensure streams have finished applying updates
  cudaDeviceSynchronize();

  // apply all outstanding deltas
  for (int stream_id = 0; stream_id < num_host_threads * stream_multiplier; stream_id++) {
    if(streams[stream_id].delta_applied == 0) {
      for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {

        for (size_t i = 0; i < num_buckets; i++) {
          delta_buckets[i].alpha = subgraphs[graph_id].cudaUpdateParams->h_bucket_a[(stream_id * num_buckets) + i];
          delta_buckets[i].gamma = subgraphs[graph_id].cudaUpdateParams->h_bucket_c[(stream_id * num_buckets) + i];
        }

        int prev_src = streams[stream_id].src_vertex;
        
        if(prev_src == -1) {
          std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
        }

        // Apply the delta sketch
        apply_raw_buckets_update((graph_id * num_nodes) + prev_src, delta_buckets);
      }
      streams[stream_id].delta_applied = 1;
      streams[stream_id].src_vertex = -1;
      streams[stream_id].num_graphs = -1;
    }
  }
}

std::vector<Edge> MCGPUSketchAlg::get_adjlist_spanning_forests() {
  return edge_store.get_edges();
}
