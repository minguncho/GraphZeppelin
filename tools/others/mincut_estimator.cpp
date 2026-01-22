#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <unordered_set>
#include <sys/resource.h> // for rusage
#include <mutex>

#include "bucket.h"
#include "dsu.h"
#include "return_types.h"
#include "util.h"

#include <algorithms/global_mincut/algorithms.h>
#include <algorithms/global_mincut/minimum_cut.h>
#include <binary_file_stream.h>
#include <data_structure/graph_access.h>
#include <data_structure/mutable_graph.h>

static constexpr size_t update_array_size = 100000;

struct MinCut {
  std::set<node_id_t> left_vertices;
  std::set<node_id_t> right_vertices;
  size_t value;
};

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}


MinCut standalone_calc_minimum_cut(node_id_t num_vertices, const std::vector<Edge> &edges) {
  typedef VieCut::mutable_graph Graph;
  typedef std::shared_ptr<VieCut::mutable_graph> GraphPtr;

  // Create a VieCut graph
  GraphPtr G = std::make_shared<Graph>();
  G->start_construction(num_vertices, edges.size());

  // Add edges to VieCut graph
  for (auto edge : edges) {
    G->new_edge(edge.src, edge.dst);
  }

  // finish construction and compute degrees
  // TODO: Don't know if degrees are necessary. Its in the VieCut code tho
  G->finish_construction();
  G->computeDegrees();

  // Perform the mincut computation
  VieCut::EdgeWeight cut;
  VieCut::minimum_cut* mc = new VieCut::viecut<GraphPtr>();
  cut = mc->perform_minimum_cut(G);

  // Return answer
  std::set<node_id_t> left;
  std::set<node_id_t> right;

  for (node_id_t i = 0; i < num_vertices; i++) {
    if (G->getNodeInCut(i))
      left.insert(i);
    else
      right.insert(i);
  }

  delete mc;
  return {left, right, cut};
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, eps" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_file = argv[1];
  double epsilon = std::stod(argv[2]);

  BinaryFileStream stream(stream_file);
  node_id_t num_vertices = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_vertices << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;

  int k = ceil(log2(num_vertices) / (epsilon * epsilon));

  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "k: " << k << std::endl;

  size_t sketch_seed = 0;
  std::cout << "Sketch Seed: " << sketch_seed << std::endl;

  // Compute max id for checking subgraphs (size of subgraph is less than its max k spanning forests)
  int max_subgraph_id = 0;
 
  for (int k_id = 0; k_id < k; k_id++) {
    size_t max_sf_size = size_t(k) * size_t(num_vertices);
    size_t num_est_edges = num_updates / (1 << k_id);

    if (num_est_edges < max_sf_size) {
      max_subgraph_id = k_id;
      break;
    }
  }
  // Note: To save space, only maintaining max_subgraph_id number of subgraphs 
  std::cout << "Max subgraph id that can have full spanning forests: " << max_subgraph_id << std::endl;

  // Setup subgraphs
  std::vector<std::vector<DisjointSetUnion_MT<node_id_t>>> subgraph_dsus;
  std::vector<std::vector<std::vector<node_id_t>*>> subgraph_k_spanning_forests;
  std::vector<std::vector<std::mutex*>> subgraph_mtxs;

  for (int graph_id = 0; graph_id <= max_subgraph_id; graph_id++) {
    std::vector<DisjointSetUnion_MT<node_id_t>> dsu;
    std::vector<std::vector<node_id_t>*> k_spanning_forests;
    std::vector<std::mutex*> mtxs;

    for (int k_id = 0; k_id < k; k_id++) {
      dsu.push_back(DisjointSetUnion_MT(num_vertices));
      k_spanning_forests.push_back(new std::vector<node_id_t>[num_vertices]);
      mtxs.push_back(new std::mutex[num_vertices]);
    }

    subgraph_dsus.push_back(dsu);
    subgraph_k_spanning_forests.push_back(k_spanning_forests);
    subgraph_mtxs.push_back(mtxs);
  }

  std::cout << "Finished initiailizing subgraphs\n";
  std::cout << "  Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;

  // Prepare for collecting edges from binary stream
  GraphStreamUpdate update_array[update_array_size];
  std::atomic<bool> read_complete = false;
  size_t total_read_updates = 0;

  std::vector<std::atomic<size_t>> subgraph_num_edges(k);

  auto timer_start = std::chrono::steady_clock::now();
  std::cout << "Collecting edges...\n";
  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    /*total_read_updates += updates;
    if (total_read_updates % 10000000 == 0) {
      std::cout << "  Progress: " << total_read_updates << "\n";
      std::cout << "Size of subgraphs:" << std::endl;
      for (int graph_id = 0; graph_id <= max_subgraph_id; graph_id++) {
        std::cout << "  G_" << graph_id << ": " << subgraph_num_edges[graph_id] << std::endl;
      }
    }*/

    #pragma omp parallel for
    for (size_t i = 0; i < updates; i++) {
      if (read_complete) continue;

      GraphStreamUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);

      // Only handle edges with src < dst
      if (upd.edge.src > upd.edge.dst) continue;

      if (upd.type == BREAKPOINT) {
        read_complete = true;
      }
      else {
        // Determine the depth of current edge
        vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(upd.edge.src, upd.edge.dst));
        int subgraph = Bucket_Boruvka::get_index_depth(edge_id, sketch_seed, k-1);

        for (int graph_id = 0; graph_id <= subgraph; graph_id++) {
          // If current subgraph's spanning forests are full, skip
          if (subgraph_num_edges[graph_id] == ((k * num_vertices) - k)) continue;

          // Graph id going out of bounds, skip
          if (graph_id > max_subgraph_id) continue;

          // Attempt to insert to k spanning forests
          bool merged = false;
          for (int k_id = 0; k_id < k; k_id++) {
            if (merged) continue;
            if (subgraph_dsus[graph_id][k_id].merge(upd.edge.src, upd.edge.dst).merged) {
              subgraph_num_edges[graph_id]++;
              {
                std::lock_guard<std::mutex> lk(subgraph_mtxs[graph_id][k_id][upd.edge.src]);
                subgraph_k_spanning_forests[graph_id][k_id][upd.edge.src].push_back(upd.edge.dst);
              }
              merged = true;
            }
          }
        }

        // Only exit if (1) Went through all the edge stream, or (2) k spanning forests for all subgraphs are full
        int num_full = 0;
        for (int graph_id = 0; graph_id <= max_subgraph_id; graph_id++) {
          if (subgraph_num_edges[graph_id] == ((k * num_vertices) - k)) num_full++;
        }
        if (num_full == max_subgraph_id + 1) {
          read_complete = true;
        }
      }
    }
  }
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished building k spanning forests for each subgraphs...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";
  std::cout << "  Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;

  std::cout << "Size of subgraphs:" << std::endl;
  for (int graph_id = 0; graph_id <= max_subgraph_id; graph_id++) {
    std::cout << "  G_" << graph_id << ": " << subgraph_num_edges[graph_id] << std::endl;
  }

  std::cout << "Computing minimum cut:" << std::endl;
  for (int graph_id = 0; graph_id <= max_subgraph_id; graph_id++) {
    if (subgraph_num_edges[graph_id] == ((k * num_vertices) - k)) {
      std::cout << "  G_" << graph_id << ": Full k spanning forests, skipping\n";
      continue;
    }

    if (subgraph_num_edges[graph_id] == 0) break;

    // Convert SFs into edge list
    std::vector<Edge> edges;
    std::unordered_set<size_t> check_edges;

    for (int k_id = 0; k_id < k; k_id++) {
      for (node_id_t src = 0; src < num_vertices; src++) {
        for (auto& dst : subgraph_k_spanning_forests[graph_id][k_id][src]) {
          size_t edge_id = static_cast<vec_t>(concat_pairing_fn(src, dst));
          if (check_edges.find(edge_id) != check_edges.end()) {
            std::cerr << "Duplicate edge found! k_id: " << k_id << " " << src << ", " << dst << "\n";
            exit(EXIT_FAILURE); 
          }

          edges.push_back({src, dst});
          check_edges.insert(edge_id);
        }
      }
    }  

    MinCut mc = standalone_calc_minimum_cut(num_vertices, edges);
    std::cout << "  G_" << graph_id << ": " << mc.value << std::endl;

    if (mc.value < k) {
      std::cout << "Mincut found in graph: " << graph_id << " mincut: " << mc.value << std::endl;
      std::cout << "Final mincut value: " << (mc.value * (pow(2, graph_id))) << std::endl;
      break;
    }
  }

  std::cout << "Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;
  
}