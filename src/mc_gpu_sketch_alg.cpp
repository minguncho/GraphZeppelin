#include "mc_gpu_sketch_alg.h"

#include <iostream>
#include <thread>
#include <vector>

void MCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (MCSketchAlg::get_update_locked()) throw UpdateLockedException();

  // If trim enabled, perform sketch updates in CPU
  if (trim_enabled) {
    if (trim_graph_id < 0 || trim_graph_id >= num_graphs) {
      std::cout << "INVALID trim_graph_id: " << trim_graph_id << "\n";
    }
    
    if (subgraphs[trim_graph_id]->get_type() != SKETCH) {
      std::cout << "Current trim_graph_id isn't SKETCH data structure: " << trim_graph_id << "\n";
    }

    batch_sizes += dst_vertices.size();
    apply_update_batch_single_graph(thr_id, trim_graph_id, src_vertex, dst_vertices);
  }

  else {

    std::vector<std::vector<node_id_t>> local_buffer;
    local_buffer.assign(num_graphs, std::vector<node_id_t>());
    int max_depth = 0;

    for (vec_t i = 0; i < dst_vertices.size(); i++) {
      // Determine the depth of current edge
      vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[i]));
      int depth = Bucket_Boruvka::get_index_depth(edge_id, 0, num_graphs-1);
      max_depth = std::max(depth, max_depth);

      for (int graph_id = 0; graph_id <= depth; graph_id++) {
        local_buffer[graph_id].push_back(dst_vertices[i]);
      }
    } 
    
    // Go every subgraph and apply updates
    for (int graph_id = 0; graph_id <= max_depth; graph_id++) {
      if (graph_id >= max_sketch_graphs) { // Fixed Adj. list
        subgraphs[graph_id]->insert_adj_edge(src_vertex, local_buffer[graph_id]);
      }
      else {
        if (subgraphs[graph_id]->get_type() == SKETCH) { // Perform Sketch updates
          subgraphs[graph_id]->insert_sketch_buffer(thr_id, src_vertex, local_buffer[graph_id]);
        }
        else { // Perform Adj. list updates
          subgraphs[graph_id]->insert_adj_edge(src_vertex, local_buffer[graph_id]);

          // Check the size of adj. list after insertion
          double adjlist_bytes = subgraphs[graph_id]->get_num_updates() * adjlist_edge_bytes;

          if (adjlist_bytes > sketch_bytes) { // With size of current adj. list, it is more space-efficient to convert into sketch graph

            if(subgraphs[graph_id]->try_acq_conversion()) {
              // Init sketches 
              std::cout << "Graph #" << graph_id << " is now sketch graph\n";

              //convert_sketch = graph_id;
              num_adj_graphs--;
              num_sketch_graphs++;

              subgraphs[graph_id]->set_type(SKETCH);
            }
          }
        }
      }
    }    
  }
}

void MCGPUSketchAlg::flush_buffers() {
  if (num_sketch_graphs == 0) {
    return;
  }

  std::cout << "Flushing buffers for (" << num_sketch_graphs << ") sketch graphs\n";
  std::vector<std::chrono::duration<double>> indiv_flush_time;
  auto flush_start = std::chrono::steady_clock::now();

  for (int graph_id = 0; graph_id < num_sketch_graphs; graph_id++) {
    auto indiv_flush_start = std::chrono::steady_clock::now();
    subgraphs[graph_id]->flush_sketch_buffers();
    cudaDeviceSynchronize();
    indiv_flush_time.push_back(std::chrono::steady_clock::now() - indiv_flush_start);
  }

  std::chrono::duration<double> flush_time = std::chrono::steady_clock::now() - flush_start;
  std::cout << "Finished flushing buffers for (" << num_sketch_graphs << ") sketch graphs. Total Elpased time: " << flush_time.count() << "\n";
  for (int graph_id = 0; graph_id < num_sketch_graphs; graph_id++) {
    std::cout << "  S" << graph_id << ": " << indiv_flush_time[graph_id].count() << "\n";
  }

}

void MCGPUSketchAlg::convert_adj_to_sketch() {
  if (num_sketch_graphs == 0) {
    return;
  }

  std::cout << "Converting adj.list graphs (" << num_sketch_graphs << ") into sketch graphs\n";
  std::vector<std::chrono::duration<double>> indiv_conversion_time;
  auto conversion_start = std::chrono::steady_clock::now();

  // Rewrite GPU Kernel Shared Memory's size
  size_t maxBytes = num_buckets * sizeof(vec_t) + num_buckets * sizeof(vec_hash_t);
  cudaKernel.updateSharedMemory(maxBytes);

  for (int graph_id = 0; graph_id < num_sketch_graphs; graph_id++) {
    std::cout << "Graph #" << graph_id << "...";
    auto indiv_conversion_start = std::chrono::steady_clock::now();

    int current_index = 0;
    int batch_count = 0;

    node_id_t *h_edgeUpdates, *d_edgeUpdates;
    gpuErrchk(cudaMallocHost(&h_edgeUpdates, subgraphs[graph_id]->get_num_adj_edges() * sizeof(node_id_t)));
    gpuErrchk(cudaMalloc(&d_edgeUpdates, subgraphs[graph_id]->get_num_adj_edges() * sizeof(node_id_t)));

    std::vector<node_id_t> h_update_src;
    std::vector<vec_t> h_update_sizes, h_update_start_index;

    for (node_id_t src = 0; src < num_nodes; src++) {
      std::set<node_id_t> dst_vertices = subgraphs[graph_id]->get_neighbor_nodes(src);
            
      if (dst_vertices.size() == 0) { // No neighbor nodes for this src vertex
        continue;
      }

      h_update_start_index.push_back(current_index);

      // Go through all neighbor nodes and fill in buffer
      for (auto dst : dst_vertices) {
        h_edgeUpdates[current_index] = dst;
        current_index++;
      }

      // Update num_sketch_updates
      subgraphs[graph_id]->increment_num_sketch_updates(dst_vertices.size());
      h_update_sizes.push_back(dst_vertices.size());
      h_update_src.push_back(src);
      batch_count++;
    } 

    node_id_t *d_update_src;
    vec_t *d_update_sizes, *d_update_start_index;
    gpuErrchk(cudaMalloc(&d_update_src, batch_count * sizeof(node_id_t)));
    gpuErrchk(cudaMalloc(&d_update_sizes, batch_count * sizeof(vec_t)));
    gpuErrchk(cudaMalloc(&d_update_start_index, batch_count * sizeof(vec_t)));

    gpuErrchk(cudaMemcpy(d_update_src, h_update_src.data(), batch_count * sizeof(node_id_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_update_sizes, h_update_sizes.data(), batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_update_start_index, h_update_start_index.data(), batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, subgraphs[graph_id]->get_num_adj_edges() * sizeof(node_id_t), cudaMemcpyHostToDevice));
    cudaKernel.single_sketchUpdate(num_device_threads, batch_count, batch_count, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, subgraphs[graph_id]->get_cudaUpdateParams(), sketchSeed);
    cudaDeviceSynchronize();

    indiv_conversion_time.push_back(std::chrono::steady_clock::now() - indiv_conversion_start);
    std::cout << "Finished.\n";
  }

  std::chrono::duration<double> conversion_time = std::chrono::steady_clock::now() - conversion_start;
  std::cout << "Finished converting adj.list graphs (" << num_sketch_graphs << ") into sketch graphs. Total Elpased time: " << conversion_time.count() << "\n";
  for (int graph_id = 0; graph_id < num_sketch_graphs; graph_id++) {
    std::cout << "  S" << graph_id << ": " << indiv_conversion_time[graph_id].count() << "\n";
  }
}

std::vector<Edge> MCGPUSketchAlg::get_adjlist_spanning_forests(int graph_id, int k) {
  if (subgraphs[graph_id]->get_type() == SKETCH) {
    std::cout << "Subgraph with graph_id: " << graph_id << " is Sketch graph!\n";
  }
  
  std::vector<Edge> edges;
  for (node_id_t src = 0; src < num_nodes; src++) {
    for (auto dst : subgraphs[graph_id]->get_neighbor_nodes(src)) {
      edges.push_back({src, dst});
    }

    // Delete sampled edge from adj. list
    //std::cout << "    Trimming spanning forest " << k_id << "\n";
    //subgraphs[graph_id]->adjlist_trim_forest(forest);
  }
  return edges;
}
