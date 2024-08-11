#include "edge_store.h"

#include <gtest/gtest.h>
#include <chrono>
#include <set>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

TEST(EdgeStoreTest, no_contract) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 20;  // artificially large
  EdgeStore edge_store(get_seed(), nodes, skt_bytes, 20);

  std::set<Edge> edges_added;

  for (size_t i = 0; i < nodes; i++) {
    std::vector<node_id_t> dsts;
    for (size_t j = 0; j < i; j++) {
      dsts.push_back(j);
      ASSERT_TRUE(edges_added.insert({std::min(i, j), std::max(i, j)}).second);
    }
    edge_store.insert_edges(i, dsts);
  }

  ASSERT_EQ(edge_store.get_num_edges(), nodes * nodes / 2 - nodes);
  ASSERT_EQ(edge_store.get_first_store_subgraph(), 0);

  std::vector<Edge> edges = edge_store.get_edges();
  ASSERT_EQ(edges.size(), nodes * nodes / 2 - nodes);

  for (auto edge : edges) {
    node_id_t src = std::min(edge.src, edge.dst);
    node_id_t dst = std::max(edge.src, edge.dst);
    ASSERT_NEQ(edges.find({src, dst}), edges.end());
  }
}
