#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <omp.h>

#include "binary_file_stream.h"
#include "util.h"

struct Subreddit {
  std::unordered_set<node_id_t> user_ids;
};

struct Interactions {
  std::vector<Subreddit> subreddit_ids;
  size_t num_users;
  size_t num_subreddits;
};

// parse the input file as a vector of interactions.
Interactions parse_file(std::string filename) {
  std::string line;
  std::ifstream myfile(filename);
  Interactions output;
  if (myfile.is_open())
  {
    //handle the header line
    getline(myfile, line);

    std::stringstream ss(line);
    std::string num_users_str;
    getline(ss, num_users_str, ',');
    std::string num_subreddits_str;
    getline(ss, num_subreddits_str, ',');
    std::string num_lines_str;
    getline(ss, num_lines_str, '\n');
    size_t num_users = std::stoull(num_users_str);
    size_t num_subreddits = std::stoull(num_subreddits_str);
    size_t num_lines = std::stoull(num_lines_str);

    printf("Dataset has %lu users, %lu subreddits, and %lu interactions \n", num_users, num_subreddits, num_lines);

    output.subreddit_ids = std::vector<Subreddit>(num_subreddits);
    output.num_users = num_users;
    output.num_subreddits = num_subreddits;

    //loop over remaining lines
    while ( getline (myfile,line) ) {
      std::stringstream ss(line);
      std::string userIDstr;
      std::string subredditIDstr;
      getline(ss, userIDstr, ',');
      getline(ss, subredditIDstr, '\n');

      output.subreddit_ids[std::stoi(subredditIDstr)].user_ids.insert(std::stoi(userIDstr));
    }
    myfile.close();
  }

  else std::cout << "Unable to open file\n"; 

  return output;
};

std::vector<size_t> collect_edges(Interactions interactions) {
  int num_threads = omp_get_max_threads();
  std::vector<std::unordered_set<size_t>> local_edges(num_threads);

  std::cout << "collect_edges() num_threads: " << num_threads << std::endl;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto& edges = local_edges[tid];

    #pragma omp for schedule(dynamic)
    for (size_t s_id = 0; s_id < interactions.subreddit_ids.size(); s_id++) {
      if (s_id % 1000 == 0) std::cout << "collected_edges() s_id: " << s_id << std::endl;

      Subreddit subreddit = interactions.subreddit_ids[s_id];
      for (auto i = subreddit.user_ids.begin(); i != subreddit.user_ids.end(); i++) {
        auto j = i;
        j++;
        for (; j != subreddit.user_ids.end(); j++) {
          node_id_t src = *i;
          node_id_t dst = *j;
          edges.insert(concat_pairing_fn(src, dst));
        }
      }
    }
  }

  std::cout << "collected_edges() Finished getting local_edges" << std::endl;

  size_t global_num_edges = 0;
  for (auto& local : local_edges) {
    global_num_edges += local.size();
  }

  std::vector<size_t> global_edges;
  global_edges.reserve(global_num_edges);

  // Insert edges
  for (auto& local : local_edges)
    global_edges.insert(global_edges.end(), local.begin(), local.end());

  // Remove any duplicate edges
  std::sort(global_edges.begin(), global_edges.end());
  global_edges.erase(std::unique(global_edges.begin(), global_edges.end()), global_edges.end());

  std::cout << "Total number of edges collected: " << global_num_edges << std::endl;
  std::cout << "Size of global_edges: " << global_edges.size() 
            << ". Duplicate edges: " << global_num_edges - global_edges.size() << std::endl;

  return global_edges;
}

void build_graph_stream(std::string filename, std::vector<size_t> edges, size_t num_nodes, size_t num_edges) {
  std::string stream_name = filename.substr(0, filename.length() - 4) + "_stream_binary";
	BinaryFileStream fout(stream_name, false);

  fout.write_header(num_nodes, num_edges);

  int num_threads = omp_get_max_threads();
  size_t update_buffer_size = 5000;
  std::vector<std::vector<GraphStreamUpdate>> local_updates(num_threads);

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto& updates = local_updates[tid];
    updates.reserve(update_buffer_size);

    #pragma omp for schedule(dynamic)
    for (size_t i = 0; i < num_edges; ++i) {
      if (i % 100000000 == 0) std::cout << "build_graph_stream() current edge it: " << i << std::endl;

      GraphStreamUpdate update;
      update.edge = inv_concat_pairing_fn(edges[i]);
      update.type = INSERT;
      updates.push_back(update);

      // Insert to binary stream
      if (updates.size() == update_buffer_size) {
        #pragma omp critical
        {
          fout.write_updates(updates.data(), updates.size());
        }
        updates.clear();
      }
    }

    // Insert the remaining to the binary stream
    if (!updates.empty()) {
      #pragma omp critical
      {
        fout.write_updates(updates.data(), updates.size());
      }
      updates.clear();
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: input_file" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto timer_start = std::chrono::steady_clock::now();
  std::string filename = argv[1];
  Interactions interactions = parse_file(filename);

  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished parsing input file: " << duration.count() << "s " <<  std::endl;

  timer_start = std::chrono::steady_clock::now();
  std::vector<size_t> edges = collect_edges(interactions);

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished collecting edges: " << duration.count() << "s " <<  std::endl;

  size_t num_nodes = interactions.num_users; 
	size_t num_edges = edges.size();

  std::cout << "Num Nodes: " << num_nodes << std::endl;
  std::cout << "Num Edges: " << num_edges << std::endl;

  // Shuffle edges 
  timer_start = std::chrono::steady_clock::now();
  std::default_random_engine e(0);
  std::shuffle(edges.begin(), edges.end(), e);

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished shuffling edges: " << duration.count() << "s " <<  std::endl;

  timer_start = std::chrono::steady_clock::now();
  build_graph_stream(filename, edges, num_nodes, num_edges);

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished converting to binary stream: " << duration.count() << "s " <<  std::endl;
}