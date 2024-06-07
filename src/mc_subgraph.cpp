#include "mc_subgraph.h"

#include <chrono>

// Constructor
MCSubgraph::MCSubgraph(int graph_id, int num_threads, CudaUpdateParams* cudaUpdateParams,
                       GraphType type, node_id_t num_nodes, double sketch_bytes,
                       double adjlist_edge_bytes)
    : graph_id(graph_id),
      num_threads(num_threads),
      cudaUpdateParams(cudaUpdateParams),
      type(type),
      num_nodes(num_nodes),
      sketch_bytes(sketch_bytes),
      adjlist_edge_bytes(adjlist_edge_bytes) {
  conversion_counter = 0;

  num_sketch_updates = 0;
  num_adj_edges = 0;

  adj_list_bufs.resize(num_threads);
}

// There is a race condition here. If one thread calls insert_adj_edges then goes to sleep 
// for an hour another thread might come along and convert the subgraph to a sketch.
void MCSubgraph::insert_adj_edges(int thr_id, node_id_t src, std::vector<node_id_t> dst_vertices) {
  if (dst_vertices.size() >= adj_list_buf_capacity) {
    // don't buffer, apply directly
    std::lock_guard<std::mutex> lk(adj_mutex);
    for (auto dst : dst_vertices) {
      Edge edge = {src, dst};
      if (adjlist.find(edge) == adjlist.end()) {
        adjlist.insert(edge);
        num_adj_edges++;
      } else {
        adjlist.erase(edge);  // Current edge already exist, so delete
        num_adj_edges--;
      }
    }
    return;
  }

  // small number of updates. Buffer these
  auto &buf = adj_list_bufs[thr_id];
  for (auto dst : dst_vertices) {
    if (buf.size >= adj_list_buf_capacity) {
      // flush
      std::lock_guard<std::mutex> lk(adj_mutex);
      for (auto edge : buf.edges) {
        if (adjlist.find(edge) == adjlist.end()) {
          adjlist.insert(edge);
          num_adj_edges++;
        } else {
          adjlist.erase(edge);  // Current edge already exist, so delete
          num_adj_edges--;
        }
      }
      buf.size = 0;
    }
    buf.edges[buf.size++] = {src, dst};
  }
}

const std::set<Edge>& MCSubgraph::get_adjlist() {
  std::lock_guard<std::mutex> lk(adj_mutex);
  for (auto &buf : adj_list_bufs) {
    if (buf.size > 0) {
      for (auto edge : buf.edges) {
        if (adjlist.find(edge) == adjlist.end()) {
          adjlist.insert(edge);
          num_adj_edges++;
        } else {
          adjlist.erase(edge);  // Current edge already exist, so delete
          num_adj_edges--;
        }
      }
    }
  }

  return adjlist;
}
