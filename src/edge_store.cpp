#include "edge_store.h"
#include "bucket.h"
#include "util.h"

#include <chrono>

// Constructor
EdgeStore::EdgeStore(size_t seed, node_id_t num_vertices, size_t sketch_bytes, size_t num_subgraphs)
    : seed(seed),
      num_vertices(num_vertices),
      num_subgraphs(num_subgraphs),
      adjlist(num_vertices),
      sketch_bytes(sketch_bytes),
      vertex_contracted(num_vertices, false) {
  num_edges = 0;
  adj_mutex = new std::mutex[num_vertices];
}

EdgeStore::~EdgeStore() {
  delete[] adj_mutex;
}

TaggedUpdateBatch EdgeStore::insert_adj_edges(node_id_t src,
                                                   const std::vector<node_id_t> &dst_vertices) {
  int edges_delta = 0;
  std::vector<SubgraphTaggedUpdate> ret;
  {
    std::lock_guard<std::mutex> lk(adj_mutex[src]);
    if (true_store_depth < store_depth && !vertex_contracted[src]) {
      ret = vertex_contract(src);
    }

    for (auto dst : dst_vertices) {
      auto idx = concat_pairing_fn(src, dst);
      SubgraphTaggedUpdate data = {Bucket_Boruvka::get_index_depth(idx, seed, num_subgraphs), dst};
      if (!adjlist[src].insert(data).second) {
        adjlist[src].erase(data);  // Current edge already exist, so delete
        edges_delta--;
      } else {
        edges_delta++;
      }
    }
  }
  num_edges += edges_delta;

  if (true_store_depth < store_depth && needs_contraction < num_vertices && ret.size() == 0) {
    return vertex_advance_subgraph();
  } else {
    check_if_too_big();
  }

  return {src, ret};
}

TaggedUpdateBatch EdgeStore::insert_adj_edges(size_t sketch_subgraphs, node_id_t src,
                                              const std::vector<SubgraphTaggedUpdate> &dst_data) {
  int edges_delta = 0;
  std::vector<SubgraphTaggedUpdate> ret;
  {
    std::lock_guard<std::mutex> lk(adj_mutex[src]);
    if (true_store_depth < store_depth && !vertex_contracted[src]) {
      ret = vertex_contract(src);
    }

    for (auto data : dst_data) {
      if (!adjlist[src].insert(data).second) {
        adjlist[src].erase(data);  // Current edge already exist, so delete
        edges_delta--;
      } else {
        edges_delta++;
      }
    }
  }
  num_edges += edges_delta;

  if (true_store_depth < store_depth && needs_contraction < num_vertices && ret.size() == 0) {
    return vertex_advance_subgraph();
  } else {
    check_if_too_big();
  }

  return {src, ret};
}

// IMPORTANT: We must have completed any pending contractions before we call this function
std::vector<Edge> EdgeStore::get_edges() {
  std::vector<Edge> ret;
  ret.resize(num_edges);

  for (node_id_t src = 0; src < num_vertices; src++) {
    for (auto data : adjlist[src])
      ret.push_back({src, data.dst});
  }

  return ret;
}

std::vector<SubgraphTaggedUpdate> EdgeStore::vertex_contract(node_id_t src) {
  std::vector<SubgraphTaggedUpdate> ret;
  ret.resize(adjlist[src].size());
  int edges_delta = 0;
  {
    std::lock_guard<std::mutex> lk(adj_mutex[src]);
    auto it_begin = adjlist[src].begin();
    auto it = it_begin;
    for (; it != adjlist[src].end(); it++) {
      if (it->subgraph >= store_depth) {
        // got to the end of stuff that should be removed
        adjlist[src].erase(it_begin, --it);
        num_edges += edges_delta;
        ++it;
      }
      ret.push_back(*it);
      edges_delta--;
    }
  }

  if (src == num_vertices - 1) {
    true_store_depth++;
  }
  return ret;
}

TaggedUpdateBatch EdgeStore::vertex_advance_subgraph() {
  node_id_t src = 0;
  do {
    src = needs_contraction.fetch_add(1);
    if (src > num_vertices) return {0, std::vector<SubgraphTaggedUpdate>()};
  } while (!vertex_contracted[src]);

  return {src, vertex_contract(src)};
}

// checks if we should perform a contraction and begins the process if so
void EdgeStore::check_if_too_big() {
  // TODO: Is sketch_bytes the size of the entire datastructure or a single sketch?
  if (num_edges * store_edge_bytes < sketch_bytes) {
    // no contraction needed
    return;
  }

  // we may need to perform a contraction
  std::lock_guard<std::mutex> lk(contract_lock);
  if (true_store_depth < store_depth) {
    // another thread already started contraction
    return;
  }

  store_depth++;
  needs_contraction = 0;
  for (node_id_t i = 0; i < num_vertices; i++) {
    vertex_contracted[i] = false;
  }
  std::cout << "EdgeStore: Contracting to subgraphs " << store_depth << " and above" << std::endl;
}
