#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "binary_file_stream.h"

static constexpr size_t update_array_size = 10000;

int main(int argc, char** argv) {

  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_file = argv[1];

  BinaryFileStream stream(stream_file);
  node_id_t num_vertices = stream.vertices();
  size_t num_updates  = stream.edges();

  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_vertices << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;

  GraphStreamUpdate update_array[update_array_size];
  std::vector<Edge> edges;
  edges.reserve(num_updates);

  size_t total_read_updates = 0;
  bool read_complete = false;
  auto timer_start = std::chrono::steady_clock::now();

  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    total_read_updates += updates;
    if (total_read_updates % 1000000000 == 0) {
      std::cout << "  Progress: " << total_read_updates << "\n";
    }

    for (size_t i = 0; i < updates; i++) {
      GraphStreamUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);
      if (upd.type == BREAKPOINT) {
        read_complete = true;
        total_read_updates--;
        break;
      }
      else {
        edges.push_back(upd.edge);
      }
    }
  }
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;

  std::cout << "Number of read edges:" << total_read_updates << "\n";
  std::cout << "Number of collected edges: " << edges.size() << "\n";

  if ((num_updates != total_read_updates) || (num_updates != edges.size())) {
    std::cout << "Mismatching collected num_updates! " << total_read_updates << ", " << edges.size() << "\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Finished Reading Input Graph File...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";

  std::string output_name = stream_file + "_edgelist.txt";
  std::ofstream output_file(output_name);

  std::cout << "Writing Edgelist file...\n";
  timer_start = std::chrono::steady_clock::now();

  output_file << num_vertices << " " << num_updates << "\n";

  for (auto e : edges) {
    output_file << e.src << " " << e.dst << "\n";
  }

  output_file.close();

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished Writing Edgelist file...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";

}