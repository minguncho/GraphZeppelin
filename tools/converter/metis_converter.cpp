#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "binary_file_stream.h"

/*int main(int argc, char** argv) {

  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string file_name = argv[1];
  std::ifstream graph_file(file_name);
  std::string line;
  
  size_t num_nodes = 0;
  size_t num_edges = 0;
  size_t current_node_id = 1;
  size_t num_self_edges = 0;

  std::map<size_t, size_t> simplified_node_ids;
  std::map<size_t, std::vector<size_t>> nodes_list;
  
  std::cout << "Input Graph File: " << file_name << "\n";
  std::cout << "Reading Input Graph File...\n";

  if(graph_file.is_open()) {
    while(std::getline(graph_file, line)) {
      std::istringstream iss(line);
      std::string token;

      size_t node1, node2;

      std::getline(iss, token, ' '); // Make sure to check delimiter
      node1 = std::stoi(token);

      std::getline(iss, token, ' '); // Make sure to check delimiter
      node2 = std::stoi(token);

      if (simplified_node_ids.find(node1) == simplified_node_ids.end()) {
        simplified_node_ids[node1] = current_node_id;
        nodes_list[current_node_id] = std::vector<size_t>();

        num_nodes++;
        current_node_id++;
      }

      if (simplified_node_ids.find(node2) == simplified_node_ids.end()) {
        simplified_node_ids[node2] = current_node_id;
        nodes_list[current_node_id] = std::vector<size_t>();

        num_nodes++;
        current_node_id++;
      }
      
      size_t simplified_node1 = simplified_node_ids[node1];
      size_t simplified_node2 = simplified_node_ids[node2];
      
      if (simplified_node1 == simplified_node2) {
        num_self_edges++;
      }
      
      nodes_list[simplified_node1].push_back(simplified_node2);
      nodes_list[simplified_node2].push_back(simplified_node1);

      num_edges++;
    }
  }
  else {
    std::cout << "Error: Couldn't find file name: " << file_name << "!\n";
  }

  std::cout << "  Num Nodes: " << num_nodes << "\n";
  std::cout << "  Num Input Edges: " << num_edges << "\n";
  std::cout << "  Num Self Edges: " << num_self_edges << "\n";

  num_edges -= num_self_edges;

  std::cout << "  Num Final Edges: " << num_edges << "\n";
  std::cout << "Finished Reading Input Graph File...\n";

  graph_file.close();

  std::string metis_name = file_name + ".metis";
  std::ofstream metis_file(metis_name);

  std::cout << "Writing METIS file...\n";

  metis_file << num_nodes << " " << num_edges << " 0" << "\n";

  for (auto it : nodes_list) {
    for (size_t neighbor = 0; neighbor < it.second.size(); neighbor++) {
      if (it.second[neighbor] == it.first) {
        continue;
      }
      metis_file << (it.second[neighbor]) << " ";
      
    }
    metis_file << "\n";  
  }
  
  metis_file.close();

  std::cout << "Finished Writing METIS file...\n";
}*/

// Converts edge stream into METIS file

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
  size_t current_node_id = 1;

  std::unordered_map<node_id_t, node_id_t> node_ids;
  std::map<node_id_t, std::vector<node_id_t>> nodes_list;
  
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
        node_id_t src = upd.edge.src;
        node_id_t dst = upd.edge.dst;

        if (node_ids.find(src) == node_ids.end()) {
          node_ids[src] = current_node_id;
          nodes_list[current_node_id] = std::vector<node_id_t>();
          current_node_id++;
        }

        if (node_ids.find(dst) == node_ids.end()) {
          node_ids[dst] = current_node_id;
          nodes_list[current_node_id] = std::vector<node_id_t>();
          current_node_id++;
        }

        nodes_list[node_ids[src]].push_back(node_ids[dst]);
        nodes_list[node_ids[dst]].push_back(node_ids[src]);
      }
    }
  }
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;

  std::cout << "Number of collected nodes: " << node_ids.size() << "\n";
  std::cout << "Number of collected edges: " << total_read_updates << "\n";

  std::cout << "Finished Reading Input Graph File...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";

  if ((num_vertices != node_ids.size()) || (num_updates != total_read_updates)) {
    std::cout << "Mismatching collected num_vertices and num_updates!\n";
    exit(EXIT_FAILURE);
  }

  std::string metis_name = stream_file + ".metis";
  std::ofstream metis_file(metis_name);

  std::cout << "Writing METIS file...\n";
  timer_start = std::chrono::steady_clock::now();

  metis_file << num_vertices << " " << num_updates << " 0" << "\n";

  for (auto it : nodes_list) {
    for (size_t neighbor = 0; neighbor < it.second.size(); neighbor++) {
      if (it.second[neighbor] == it.first) {
        continue;
      }
      metis_file << (it.second[neighbor]) << " ";
      
    }
    metis_file << "\n";  
  }
  
  metis_file.close();

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished Writing METIS file...\n";
  std::cout << "  Duration: " << duration.count() << "s\n";

}