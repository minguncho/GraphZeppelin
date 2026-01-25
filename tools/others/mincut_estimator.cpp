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
#include <omp.h>

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

void init_subgraph(int graph_id, int k, node_id_t num_vertices, 
                   std::vector<std::vector<DisjointSetUnion_MT<node_id_t>>> &subgraph_dsus,
                   std::vector<std::vector<std::vector<node_id_t>*>> &subgraph_k_spanning_forests,
                   std::vector<std::vector<std::mutex*>> &subgraph_mtxs) {

  std::vector<DisjointSetUnion_MT<node_id_t>> dsu;
  std::vector<std::vector<node_id_t>*> k_spanning_forests;
  std::vector<std::mutex*> mtxs;

  for (int k_id = 0; k_id < k; k_id++) {
    dsu.push_back(DisjointSetUnion_MT(num_vertices));
    k_spanning_forests.push_back(new std::vector<node_id_t>[num_vertices]);
    mtxs.push_back(new std::mutex[num_vertices]);
  }

  subgraph_dsus[graph_id] = dsu;
  subgraph_k_spanning_forests[graph_id] = k_spanning_forests;
  subgraph_mtxs[graph_id] = mtxs;

  std::cout << "    + Initialized subgraph #" << graph_id << std::endl;
}

void delete_subgraph(int graph_id,
                     std::vector<std::vector<DisjointSetUnion_MT<node_id_t>>> &subgraph_dsus,
                     std::vector<std::vector<std::vector<node_id_t>*>> &subgraph_k_spanning_forests,
                     std::vector<std::vector<std::mutex*>> &subgraph_mtxs) {

  subgraph_dsus[graph_id].clear();

  for (auto* forest_ptr : subgraph_k_spanning_forests[graph_id]) {
    delete[] forest_ptr; 
  }
  subgraph_k_spanning_forests[graph_id].clear();

  for (auto* mtx_ptr : subgraph_mtxs[graph_id]) {
    delete[] mtx_ptr;
  }
  subgraph_mtxs[graph_id].clear();

  std::cout << "    - Deleted subgraph #" << graph_id << std::endl;
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

  // Counters for keeping the currrent min and max subgraph id
  int cur_min_subgraph_id = 0; // Increment when delete subgraph
  std::atomic<int> cur_max_subgraph_id = 0; // Increment when add a new subgraph

  // Compute number of full subgraphs (size of subgraph is less than its max k spanning forests)
  int num_full_subgraphs = 0;

  for (int k_id = 0; k_id < k; k_id++) {
    size_t max_sf_size = (size_t(k) * size_t(num_vertices)) - size_t(k);
    size_t num_est_edges = (num_updates / (1 << k_id));

    if (num_est_edges < max_sf_size) {
      num_full_subgraphs = k_id + 1;
      break;
    }
    
    cur_min_subgraph_id = k_id;
    cur_max_subgraph_id = k_id;
  }

  // cur_min_subgraph_id is id before subgraph gets fewer edges than full DSU
  // just to be safe, start one id before
  cur_min_subgraph_id--;
  cur_max_subgraph_id--;

  // num_full_subgraphs serve as a maximum number of disjoint set to maintain
  std::cout << "Number of subgraphs that can have full spanning forests: " << num_full_subgraphs << std::endl;
  std::cout << "Starting subgraph ID: " << cur_min_subgraph_id << std::endl;

  // Main data structure: DSUs and k spanning forests for each subgraph
  std::vector<std::vector<DisjointSetUnion_MT<node_id_t>>> subgraph_dsus(num_full_subgraphs);
  std::vector<std::vector<std::vector<node_id_t>*>> subgraph_k_spanning_forests(num_full_subgraphs);
  std::vector<std::vector<std::mutex*>> subgraph_mtxs(num_full_subgraphs);

  // Initialize first subgraph
  init_subgraph(cur_min_subgraph_id, k, num_vertices, subgraph_dsus, subgraph_k_spanning_forests, subgraph_mtxs);

  // For maintaining safe initialization of new subgraph
  std::vector<std::mutex> subgraph_init_mtxs(num_full_subgraphs);
  std::vector<std::atomic<bool>> subgraph_init_flag(num_full_subgraphs);

  for (auto& flag : subgraph_init_flag) {
    flag = false;
  }

  int num_threads = omp_get_max_threads();
  std::cout << "Number of CPU threads: " << num_threads << std::endl;
  std::vector<std::chrono::duration<double>> threads_merge_time(num_threads);

  for (int tid = 0; tid < num_threads; tid++) {
    threads_merge_time[tid] = std::chrono::nanoseconds::zero();
  }

  // Prepare for collecting edges from binary stream
  GraphStreamUpdate update_array[update_array_size];
  std::atomic<bool> read_complete = false;

  std::vector<std::atomic<size_t>> subgraph_num_edges(k);

  auto timer_start = std::chrono::steady_clock::now();
  size_t num_read_updates = 0;
  std::cout << "Collecting edges...\n";
  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    num_read_updates += updates;
    if (num_read_updates % 1000000000 == 0) {
      std::cout << "  Processed updates: " << num_read_updates << std::endl;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < updates; i++) {
      if (read_complete) continue;

      GraphStreamUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);

      if (upd.type == BREAKPOINT) {
        read_complete = true; // Finished reading the entire stream
      }
      else {
        // Determine the depth of current edge
        vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(upd.edge.src, upd.edge.dst));
        int subgraph = Bucket_Boruvka::get_index_depth(edge_id, sketch_seed, k-1);

        for (int graph_id = 0; graph_id <= subgraph; graph_id++) {
          // Trying to add to previously existed subgraph, skip
          if (graph_id < cur_min_subgraph_id) continue;

          if (graph_id > cur_max_subgraph_id) {
            // If trying to go over hard maximum, break.
            if (graph_id >= num_full_subgraphs) break;

            // In bound, just need to initialize a new subgraph
            if (!subgraph_init_flag[graph_id]) {
              std::lock_guard<std::mutex> lk(subgraph_init_mtxs[graph_id]);
              {
                // Check if current subgraph has already been initialized
                if (!subgraph_init_flag[graph_id]) {
                  init_subgraph(graph_id, k, num_vertices, subgraph_dsus, subgraph_k_spanning_forests, subgraph_mtxs);
                  cur_max_subgraph_id++;
                  subgraph_init_flag[graph_id] = true;
                }
              }
            }
          }

          // Within bound, make an attempt to insert to DSUs.
          auto merge_start = std::chrono::steady_clock::now();
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
          threads_merge_time[omp_get_thread_num()] += (std::chrono::steady_clock::now() - merge_start);
        }
      }
    }

    // Check for sizes of each subgraph, if full, delete
    for (int graph_id = cur_min_subgraph_id; graph_id <= cur_max_subgraph_id; graph_id++) {
      if (subgraph_num_edges[graph_id] == ((k * num_vertices) - k)) {
        delete_subgraph(graph_id, subgraph_dsus, subgraph_k_spanning_forests, subgraph_mtxs);
        cur_min_subgraph_id++;
      }
    }
  }
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished building k spanning forests for each subgraphs...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";
  std::cout << "  Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;

  std::cout << "Printing merge time:" << std::endl;

  for (int tid = 0; tid < num_threads; tid++) {
    std::cout << "  T" << tid << ": " << threads_merge_time[tid].count() << "s" << std::endl;
  }

  std::cout << "Sizes of currently maintaining subgraphs:" << std::endl;
  for (int graph_id = cur_min_subgraph_id; graph_id <= cur_max_subgraph_id; graph_id++) {
    std::cout << "  G_" << graph_id << ": " << subgraph_num_edges[graph_id] << std::endl;
  }

  std::cout << "Computing minimum cut:" << std::endl;
  for (int graph_id = cur_min_subgraph_id; graph_id <= cur_max_subgraph_id; graph_id++) {

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