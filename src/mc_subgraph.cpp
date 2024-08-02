#include "mc_subgraph.h"

#include <chrono>

// Constructor
MCSubgraph::MCSubgraph(int graph_id, node_id_t num_nodes, int num_host_threads, int num_device_threads, int num_batch_per_buffer,
                       CudaUpdateParams* cudaUpdateParams, GraphType type, size_t batch_size, size_t sketchSeed)
    : graph_id(graph_id),
      num_host_threads(num_host_threads),
      cudaUpdateParams(cudaUpdateParams),
      type(type),
      num_nodes(num_nodes),
      adjlist(num_nodes) {
  conversion_counter = 0;

  num_sketch_updates = 0;
  num_adj_edges = 0;

  adj_mutex = new std::mutex[num_nodes];

  // Initialize CUDA Streams
  cudaStreams = new CudaStream*[num_host_threads];
  for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
    cudaStreams[thr_id] = new CudaStream(num_device_threads, num_batch_per_buffer, batch_size, cudaUpdateParams, sketchSeed);
  }
}

MCSubgraph::MCSubgraph(int graph_id, node_id_t num_nodes, int num_host_threads, GraphType type)
    : graph_id(graph_id),
      num_host_threads(num_host_threads),
      type(type),
      num_nodes(num_nodes),
      adjlist(num_nodes) {

  num_adj_edges = 0;

  adj_mutex = new std::mutex[num_nodes];
}

MCSubgraph::~MCSubgraph() {
  delete[] adj_mutex;
}

void MCSubgraph::insert_adj_edge(node_id_t src, std::vector<node_id_t> dst_vertices) {
  std::lock_guard<std::mutex> lk(adj_mutex[src]);
  int num_updated_edges = 0;
  for (auto dst : dst_vertices) {
    if (adjlist[src].find(dst) == adjlist[src].end()) {
      adjlist[src].insert(dst);
      num_updated_edges++;
    } else {
      adjlist[src].erase(dst);  // Current edge already exist, so delete
      num_updated_edges--;
    }
  }
  num_adj_edges += num_updated_edges;
}

void MCSubgraph::insert_sketch_buffer(int thr_id, node_id_t src, std::vector<node_id_t> dst_vertices) {
  num_sketch_updates += dst_vertices.size();
  cudaStreams[thr_id]->process_batch(src, dst_vertices);
}

void MCSubgraph::flush_sketch_buffers() {
  for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
    cudaStreams[thr_id]->flush_buffers();
  }
}