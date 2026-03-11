#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <parallel/algorithm>

#include "binary_file_stream.h"

// NOTE: kron_16 and kron_17 contains edges between isolated vertices.
// Minimum cut value is still correct, but the number of minimum cuts may be different.

static constexpr size_t update_array_size = 100000;

int main(int argc, char** argv) {

  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file exact_min_cut collect_edges" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_file = argv[1];
  int exact_min_cut = std::atoi(argv[2]);
  bool collect_edges;
  if (std::string(argv[3]) == "true") {
    collect_edges = true;
  }
  else if (std::string(argv[3]) == "false") {
    collect_edges = false;
  }  
  else {
    std::cout << "Invalid option for collect_edges: " << argv[3] << ". Must be 'true' or 'false'\n";
    exit(EXIT_FAILURE); 
  }

  BinaryFileStream stream(stream_file);
  node_id_t num_vertices = stream.vertices();
  size_t num_updates  = stream.edges();

  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_vertices << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << "Collecting Edges: " << collect_edges << std::endl;

  GraphStreamUpdate update_array[update_array_size];

  std::vector<size_t> nodes_num_updates(num_vertices);
  std::vector<std::vector<node_id_t>> nodes_edges(num_vertices);

  size_t num_processed_updates = 0;
  bool read_complete = false;
  auto timer_start = std::chrono::steady_clock::now();
  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    for (size_t i = 0; i < updates; i++) {
      GraphStreamUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);
      if (upd.type == BREAKPOINT) {
        read_complete = true;
        break;
      }
      else {
        node_id_t src = upd.edge.src;
        node_id_t dst = upd.edge.dst;

        nodes_num_updates[src]++;
        nodes_num_updates[dst]++;

        if (collect_edges) {
          nodes_edges[src].push_back(dst);  
          nodes_edges[dst].push_back(src);          
        }

      }
    }

    num_processed_updates += updates;
    if (num_processed_updates % 1000000000 == 0) std::cout << "  Progress: " << num_processed_updates << "\n";

  }
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;

  std::cout << "Finished Reading Input Graph File...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";

  size_t total_read_updates = 0;

  for (auto& num_read_updates : nodes_num_updates) {
    total_read_updates += num_read_updates;
  }

  if (total_read_updates != (num_updates * 2)) {
    std::cout << "Mismatching num_updates! " << total_read_updates << " != " << num_updates << "\n";
    exit(EXIT_FAILURE);
  }

  if (collect_edges) {
    std::atomic<size_t> test_num_mincut = 0;
    // Sort to find duplicate edges
    timer_start = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (node_id_t src = 0; src < num_vertices; src++) {
      std::vector<node_id_t>& edges = nodes_edges[src];

      if (edges.empty()) continue;

      // Sort the edges
      std::sort(edges.begin(), edges.end());

      size_t writeIdx = 0;
      size_t n = edges.size();

      for (size_t i = 0; i < n; ) {
        size_t j = i;
        while (j < n && edges[i] == edges[j]) j++;

        if ((j - i) % 2 != 0) {
          edges[writeIdx++] = edges[i];
        }
        i = j;
      }
      
      if (writeIdx < n) { // Only remove if there are duplicate edges
        edges.resize(writeIdx);
      }
    }

    duration = std::chrono::steady_clock::now() - timer_start;
    std::cout << "Finished Removing Duplicate Edges...\n";
    std::cout << "  Duration: " << duration.count() << "s\n";
  }

  int num_mincut = 0;
  double total_edges = 0;
  size_t min_deg = num_vertices;
  size_t max_deg = 0;
  double avg_deg = 0;
  double sd = 0;

  std::vector<node_id_t> src_cuts;
  
  if (collect_edges) {
    for (auto& edges : nodes_edges) {
      if (edges.size() == exact_min_cut) num_mincut++;
      min_deg = std::min(min_deg, edges.size());
      max_deg = std::max(max_deg, edges.size());
      total_edges += edges.size();
    }

    avg_deg = total_edges / num_vertices;
    for (auto& edges : nodes_edges) {
      sd += std::pow(edges.size() - avg_deg, 2);
    }
    sd = std::sqrt(sd / num_vertices);

  }
  else {
    for (auto& num_read_updates : nodes_num_updates) {
      if (num_read_updates == exact_min_cut) num_mincut++;
      min_deg = std::min(min_deg, num_read_updates);
      max_deg = std::max(max_deg, num_read_updates);
      total_edges += num_read_updates;
    }

    avg_deg = total_edges / num_vertices;
    for (auto& num_read_updates : nodes_num_updates) {
      sd += std::pow(num_read_updates - avg_deg, 2);
    }
    sd = std::sqrt(sd / num_vertices);
  }

  std::cout << "-----RESULT-----\n";
  std::cout << "Exact minimum cut value: " << exact_min_cut << "\n";
  std::cout << "Number of mincuts: " << num_mincut << "\n"; 
  std::cout << "Minimum Deg: " << min_deg << "\n";
  std::cout << "Maximum Deg: " << max_deg << "\n";
  std::cout << "Average Deg: " << avg_deg << "\n";
  std::cout << "Standard Deviation of Deg: " << sd << "\n";
}