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
  volatile size_t cur_subgraph = 0; // subgraph depth at which edges enter the edge store
  volatile size_t true_min_subgraph = 0; // the minimum subgraph of elements in the store

  std::atomic<edge_id_t> num_edges;
  std::atomic<node_id_t> needs_contraction;

  std::vector<std::set<SubgraphTaggedUpdate>> adjlist;

  std::vector<bool> vertex_contracted;

  std::atomic<int> num_inserted;
  std::atomic<int> num_duplicate;
  std::atomic<int> num_returned;
  std::atomic<int> num_contracted;

  size_t sketch_bytes;        // Bytes of sketch graph
  static constexpr size_t store_edge_bytes = sizeof(SubgraphTaggedUpdate);  // Bytes of one edge

  std::mutex* adj_mutex;
  std::mutex contract_lock;

  std::vector<SubgraphTaggedUpdate> vertex_contract(node_id_t src);
  void check_if_too_big();

  void verify_contract_complete();
 public:

  // Constructor
  EdgeStore(size_t seed, node_id_t num_vertices, size_t sketch_bytes, size_t num_subgraphs);
  ~EdgeStore();

  // functions for adding data to the edge store
  // may return a vector of edges that need to be applied to

  // this first function is only called when there exist no sketch subgraphs
  TaggedUpdateBatch insert_adj_edges(node_id_t src, const std::vector<node_id_t>& dst_vertices);

  // this function is called when there are some sketch subgraphs.
  TaggedUpdateBatch insert_adj_edges(node_id_t src,
                                     const std::vector<SubgraphTaggedUpdate>& dst_data);

  // contract vertex data by removing all updates bound for lower subgraphs than the store 
  // is responsible for
  TaggedUpdateBatch vertex_advance_subgraph();

  // Get methods
  size_t get_num_edges() {
    return num_edges;
  }
  size_t get_footprint() {
    return num_edges * store_edge_bytes;
  }
  size_t get_first_store_subgraph() {
    return cur_subgraph;
  }
  std::vector<Edge> get_edges();
  bool contract_in_progress() { return true_min_subgraph < cur_subgraph; }

  void stats() {
    std::cout << "== EdgeStore Statistics ==" << std::endl;
    std::cout << "  num_inserted = " << num_inserted << std::endl;
    std::cout << "  num_duplicate = " << num_duplicate << std::endl;
    std::cout << "  num_returned = " << num_returned << std::endl;
    std::cout << "  num_contracted = " << num_contracted << std::endl;

    size_t sum = 0;
    for (node_id_t i = 0; i < num_vertices; i++) {
      sum += adjlist[i].size();
    }
    std::cout << "  Total number of edges = " << sum << std::endl;
  }
};
