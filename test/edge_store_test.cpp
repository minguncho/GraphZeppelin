#include "edge_store.h"
#include "bucket.h"
#include "util.h"

#include <gtest/gtest.h>
#include <chrono>
#include <set>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

TEST(EdgeStoreTest, no_contract) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 25;  // artificially large
  EdgeStore edge_store(get_seed(), nodes, skt_bytes, 20);

  std::set<Edge> edges_added;

  for (size_t i = 0; i < nodes; i++) {
    std::vector<node_id_t> dsts;
    for (size_t j = 0; j < i; j++) {
      dsts.push_back(j);
      ASSERT_TRUE(edges_added.insert({std::min(i, j), std::max(i, j)}).second);
    }
    auto more_upds = edge_store.insert_adj_edges(i, dsts);
    ASSERT_EQ(more_upds.dsts_data.size(), 0);
  }

  ASSERT_EQ(edge_store.get_num_edges(), edges_added.size());
  ASSERT_EQ(edge_store.get_first_store_subgraph(), 0);

  std::vector<Edge> edges = edge_store.get_edges();
  ASSERT_EQ(edges.size(), edges_added.size());

  for (auto edge : edges) {
    node_id_t src = std::min(edge.src, edge.dst);
    node_id_t dst = std::max(edge.src, edge.dst);
    ASSERT_NE(edges_added.find({src, dst}), edges_added.end());
  }
}

TEST(EdgeStoreTest, contract) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 20; // small enough to likely contract twice
  size_t num_subgraphs = 20;
  EdgeStore edge_store(get_seed(), nodes, skt_bytes, num_subgraphs);

  std::set<Edge> edges_added;

  size_t num_returned[6] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < nodes; i++) {
    std::vector<SubgraphTaggedUpdate> dsts;
    for (size_t j = 0; j < i; j++) {
      node_id_t src = std::min(i, j);
      node_id_t dst = std::max(i, j);
      auto idx = concat_pairing_fn(src, dst);
      size_t depth = Bucket_Boruvka::get_index_depth(idx, seed, num_subgraphs);
      if (depth < edge_store.get_first_store_subgraph()) {
        for (size_t k = 0; k <= depth; k++) {
          ++num_returned[k];
        }
      }
      else {
        dsts.push_back({depth, j});
      }
      ASSERT_TRUE(edges_added.insert({src, dst}).second);
    }
    auto more_upds = edge_store.insert_adj_edges(i, dsts);
    node_id_t src = more_upds.src;
    for (auto dst_data : more_upds.dsts_data) {
      ++num_returned[edge_store.get_first_store_subgraph() - 1];
      node_id_t s = std::min(src, dst_data.dst);
      node_id_t d = std::max(src, dst_data.dst);
      ASSERT_NE(edges_added.find({s, d}), edges_added.end());
    }
  }

  ASSERT_EQ(num_returned[0], edges_added.size());
  ASSERT_EQ(edge_store.get_first_store_subgraph(), 0);

  std::vector<Edge> edges = edge_store.get_edges();
  ASSERT_EQ(edges.size(), edges_added.size());

  for (auto edge : edges) {
    node_id_t src = std::min(edge.src, edge.dst);
    node_id_t dst = std::max(edge.src, edge.dst);
    ASSERT_NE(edges_added.find({src, dst}), edges_added.end());
  }
}
