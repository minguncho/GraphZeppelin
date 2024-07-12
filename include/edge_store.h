#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <vector>

#include "types.h"

class EdgeStore {
 private:
  size_t seed;
  node_id_t num_vertices;
  size_t num_subgraphs;
  volatile size_t store_depth = 0; // subgraph depth at which edges enter the edge store
  volatile size_t true_store_depth = 0; // the minimum subgraph of elements in the store

  std::atomic<edge_id_t> num_edges;
  std::atomic<node_id_t> needs_contraction;

  std::vector<std::set<SubgraphTaggedUpdate>> adjlist;

  std::vector<bool> vertex_contracted;

  size_t sketch_bytes;        // Bytes of sketch graph
  static constexpr size_t store_edge_bytes = sizeof(SubgraphTaggedUpdate);  // Bytes of one edge

  std::mutex* adj_mutex;
  std::mutex contract_lock;

  TaggedUpdateBatch vertex_advance_subgraph();
  void check_if_too_big();
 public:

  // Constructor
  EdgeStore(size_t seed, node_id_t num_vertices, size_t sketch_bytes, size_t num_subgraphs);
  ~EdgeStore();

  // functions for adding data to the edge store
  // may return a vector of edges that need to be applied to

  // this first function is only called when there exist no sketch subgraphs
  TaggedUpdateBatch insert_adj_edges(node_id_t src, const std::vector<node_id_t>& dst_vertices);

  // this function is called when there are some sketch subgraphs.
  // The caller passes in the number of subgraphs that were sketched when they built the dst_data
  TaggedUpdateBatch insert_adj_edges(size_t sketch_subgraphs, node_id_t src,
                                     const std::vector<SubgraphTaggedUpdate>& dst_data);

  // Get methods
  size_t get_num_edges() {
    return num_edges;
  }
  size_t get_footprint() {
    return num_edges * store_edge_bytes;
  }
  size_t get_first_store_subgraph() {
    return store_depth;
  }
  std::vector<Edge> get_edges();
};
