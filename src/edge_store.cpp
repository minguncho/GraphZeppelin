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
      vertex_contracted(num_vertices, true),
      sketch_bytes(sketch_bytes) {
  num_edges = 0;
  adj_mutex = new std::mutex[num_vertices];

  num_inserted = 0;
  num_duplicate = 0;
  num_returned = 0;
  num_contracted = 0;
}

EdgeStore::~EdgeStore() {
  delete[] adj_mutex;
}

TaggedUpdateBatch EdgeStore::insert_adj_edges(node_id_t src,
                                                   const std::vector<node_id_t> &dst_vertices) {
  int edges_delta = 0;
  std::vector<SubgraphTaggedUpdate> ret;
  num_inserted += dst_vertices.size();
  {
    std::lock_guard<std::mutex> lk(adj_mutex[src]);
    if (true_min_subgraph < cur_subgraph && !vertex_contracted[src]) {
      ret = vertex_contract(src);
    }

    for (auto dst : dst_vertices) {
      auto idx = concat_pairing_fn(src, dst);
      SubgraphTaggedUpdate data = {Bucket_Boruvka::get_index_depth(idx, seed, num_subgraphs), dst};

      if (data.subgraph < cur_subgraph) {
        ret.push_back(data);
        num_returned++;
      } else {
        if (!adjlist[src].insert(data).second) {
          // Current edge already exist, so delete
          if (adjlist[src].erase(data) == 0) {
            std::cerr << "ERROR: We found a duplicate but couldn't remove???" << std::endl;
            exit(EXIT_FAILURE);
          }
          edges_delta--;
          num_duplicate++;
        } else {
          edges_delta++;
        }
      }
    }
  }
  num_edges += edges_delta;

  if (ret.size() == 0 && true_min_subgraph < cur_subgraph && needs_contraction < num_vertices) {
    return vertex_advance_subgraph();
  } else {
    check_if_too_big();
    return {src, ret};
  }
}

TaggedUpdateBatch EdgeStore::insert_adj_edges(node_id_t src,
                                              const std::vector<SubgraphTaggedUpdate> &dst_data) {
  int edges_delta = 0;
  std::vector<SubgraphTaggedUpdate> ret;
  num_inserted += dst_data.size();
  {
    std::lock_guard<std::mutex> lk(adj_mutex[src]);
    if (true_min_subgraph < cur_subgraph && !vertex_contracted[src]) {
      ret = vertex_contract(src);
    }

    for (auto data : dst_data) {
      if (data.subgraph < cur_subgraph) {
        ret.push_back(data);
        num_returned++;
      } else {
        if (!adjlist[src].insert(data).second) {
          // Current edge already exist, so delete
          if (adjlist[src].erase(data) == 0) {
            std::cerr << "ERROR: We found a duplicate but couldn't remove???" << std::endl;
            exit(EXIT_FAILURE);
          }
          edges_delta--;
          num_duplicate++;
        } else {
          edges_delta++;
        }
      }
    }
  }
  num_edges += edges_delta;

  if (ret.size() == 0 && true_min_subgraph < cur_subgraph && needs_contraction < num_vertices) {
    return vertex_advance_subgraph();
  } else {
    check_if_too_big();
    return {src, ret};
  }
}

// IMPORTANT: We must have completed any pending contractions before we call this function
std::vector<Edge> EdgeStore::get_edges() {
  std::vector<Edge> ret;
  ret.reserve(num_edges);

  for (node_id_t src = 0; src < num_vertices; src++) {
    for (auto data : adjlist[src])
      ret.push_back({src, data.dst});
  }

  return ret;
}

void EdgeStore::verify_contract_complete() {
  for (size_t i = 0; i < num_vertices; i++) {
    std::lock_guard<std::mutex> lk(adj_mutex[i]);
    if (adjlist[i].size() == 0) continue;

    auto it = adjlist[i].begin();
    if (it->subgraph < cur_subgraph) {
      std::cerr << "ERROR: Found " << it->subgraph << ", " << it->dst << " which should have been deleted by contraction to " << cur_subgraph << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Contraction verified!" << std::endl;
  stats();
}

// the thread MUST hold the lock on src before calling this function
std::vector<SubgraphTaggedUpdate> EdgeStore::vertex_contract(node_id_t src) {
  std::vector<SubgraphTaggedUpdate> ret;
  // someone already contacted this vertex
  if (vertex_contracted[src])
    return ret;

  std::cerr << "Contracting vertex: " << src << " ";
  vertex_contracted[src] = true;

  if (adjlist[src].size() == 0) {
    std::cerr << "empty" << std::endl;
    return ret;
  }

  ret.reserve(adjlist[src].size());
  int edges_delta = 0;
  auto it_begin = adjlist[src].begin();
  auto it = it_begin;
  auto delete_it = it_begin;
  for (; it != adjlist[src].end(); it++) {
    if (it->subgraph < cur_subgraph) {
      delete_it++;
      edges_delta--;
    }
    ret.push_back(*it);
  }

  num_contracted -= edges_delta;

  // now perform the deletion
  adjlist[src].erase(it_begin, delete_it);
  num_edges += edges_delta;
  std::cerr << edges_delta * - 1 << " updates deleted ";

  std::cerr << ret.size() << " updates returned" << std::endl;
  num_returned += ret.size();
  return ret;
}

TaggedUpdateBatch EdgeStore::vertex_advance_subgraph() {
  node_id_t src = 0;
  do {
    src = needs_contraction.fetch_add(1);
    if (src == num_vertices - 1) {
      std::lock_guard<std::mutex> lk(contract_lock);
      std::cerr << "Got source = " << src << " incrementing true_min_subgraph" << std::endl;
      verify_contract_complete();
      ++true_min_subgraph;
    }

    if (src >= num_vertices) return {0, std::vector<SubgraphTaggedUpdate>()};
  } while (vertex_contracted[src]);

  std::lock_guard<std::mutex> lk(adj_mutex[src]);
  return {src, vertex_contract(src)};
}

// checks if we should perform a contraction and begins the process if so
void EdgeStore::check_if_too_big() {
  if (num_edges * store_edge_bytes < sketch_bytes) {
    // no contraction needed
    return;
  }

  // we may need to perform a contraction
  std::lock_guard<std::mutex> lk(contract_lock);
  if (true_min_subgraph < cur_subgraph) {
    // another thread already started contraction
    return;
  }

  std::cerr << "true_min = " << true_min_subgraph << " cur = " << cur_subgraph << std::endl;

  for (node_id_t i = 0; i < num_vertices; i++) {
    vertex_contracted[i] = false;
  }
  needs_contraction = 0;
  cur_subgraph++;

  std::cout << "EdgeStore: Contracting to subgraphs " << cur_subgraph << " and above" << std::endl;
  std::cout << "    num_edges = " << num_edges << std::endl;
  std::cout << "    store_edge_bytes = " << store_edge_bytes << std::endl; 
  std::cout << "    sketch_bytes = " << sketch_bytes << std::endl;

  stats();
}
