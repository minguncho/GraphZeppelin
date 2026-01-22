#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include "types.h"
#include <binary_file_stream.h>

static constexpr size_t update_array_size = 100000;

// If binary stream is small enough to fit in RAM, do everything in RAM.
void inplace_shuffle(std::string stream_name) {
  std::string output_name = stream_name + "_shuffled";

  /*
  * Reading Input Edge Stream
  */

  std::cout << "Processing stream: " << stream_name << std::endl;
  
  BinaryFileStream stream(stream_name);
  size_t num_updates  = stream.edges();

  std::vector<Edge> edges;
  size_t total_read_updates = 0;
  bool read_complete = false;
  GraphStreamUpdate update_array[update_array_size];

  auto timer_start = std::chrono::steady_clock::now();
  edges.resize(num_updates, {0, 0});
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Finished initializing buffer for edges: " << duration.count() << "s " <<  std::endl;

  std::cout << "Reading edges...\n";
  timer_start = std::chrono::steady_clock::now();

  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    #pragma omp parallel for
    for (size_t i = 0; i < updates; i++) {
      if (read_complete) continue;

      GraphStreamUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);

      if (upd.type == BREAKPOINT) {
        read_complete = true;
      }
      else {
        edges[total_read_updates + i] = upd.edge;
      }
    }
    total_read_updates += updates;
  }
  // Decrement 1 for break point
  total_read_updates--;

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Total number of edges read: " << total_read_updates << "\n";
  std::cout << "Reading time (sec): " << duration.count() << "\n";

  /*
   * Shuffling Edges 
   */

  std::cout << "Shuffling edges...\n";

  timer_start = std::chrono::steady_clock::now();
  std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::shuffle(edges.begin(), edges.end(), rng);

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Shuffling time (sec): " << duration.count() << "\n";

  /*
   * Writing Edge Stream with Shuffled Edges
   */

  timer_start = std::chrono::steady_clock::now();
  std::cout << "Writing edges back to stream...\n";
  BinaryFileStream fout(output_name , false);
  fout.write_header(stream.vertices(), stream.edges());

  // Note: Writing back in parallel doesn't give performance improvement
  for (size_t i = 0; i < update_array_size; i++) {
    update_array[i].edge = {0, 0};
    update_array[i].type = INSERT;
  }
  
  size_t update_array_index = 0;
  size_t written_edges = 0;
  timer_start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < edges.size(); i++) {
    update_array[update_array_index].edge = edges[i];

    update_array_index++;
    written_edges++;
    if (update_array_index == update_array_size) {
      fout.write_updates(update_array, update_array_size);
      update_array_index = 0;
    }
  }

  // Write any remaining edges to stream
  if (update_array_index > 0) {
		fout.write_updates(update_array, update_array_index);
	}

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Writing time (sec): " << duration.count() << "\n";
  std::cout << "Total number of edges written to stream: " << edges.size() << "\n"; 
}

// Binary stream is too big to fit in RAM, split into two then shuffle them separately
// to merge them together eventually.
void partition_shuffle(std::string stream_name) {
  std::string output_name = stream_name + "_shuffled";

  std::cout << "Processing stream: " << stream_name << std::endl;
  
  BinaryFileStream stream(stream_name);
  size_t num_updates = stream.edges();

  size_t first_buffer_size = ((num_updates / 2) / update_array_size) * update_array_size;
  size_t second_buffer_size = num_updates - first_buffer_size;

  std::cout << "Size of first half buffer:  " << first_buffer_size << "\n";
  std::cout << "Size of second half buffer: " << second_buffer_size << "\n";

  std::vector<Edge> edges;
  size_t total_read_updates = 0;
  bool read_complete = false;
  GraphStreamUpdate update_array[update_array_size];

  /*
  * Part 1: Read first half of binary stream, shuffle, then write
  */

  std::cout << "Processing first half:\n";

  // Initialize
  auto timer_start = std::chrono::steady_clock::now();
  edges.resize(first_buffer_size, {0, 0});
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Finished initializing buffer for edges: " << duration.count() << "s " <<  std::endl;

  // Reading edges
  std::cout << "  Reading edges...\n";
  timer_start = std::chrono::steady_clock::now();

  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    #pragma omp parallel for
    for (size_t i = 0; i < updates; i++) {
      edges[total_read_updates + i] = update_array[i].edge;
    }
    total_read_updates += updates;

    if (total_read_updates == first_buffer_size) {
      read_complete = true;
    }
  }

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "    Total number of edges read: " << total_read_updates << "\n";
  std::cout << "  Reading time (sec): " << duration.count() << "\n";

  // Shuffling edges
  std::cout << "  Shuffling edges...\n";
  timer_start = std::chrono::steady_clock::now();
  std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::shuffle(edges.begin(), edges.end(), rng);

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Shuffling time (sec): " << duration.count() << "\n";

  // Writing edges
  timer_start = std::chrono::steady_clock::now();
  std::cout << "  Writing edges back to stream...\n";
  BinaryFileStream fout_part1(output_name + "_part1", false);
  fout_part1.write_header(stream.vertices(), first_buffer_size);

  for (size_t i = 0; i < update_array_size; i++) {
    update_array[i].edge = {0, 0};
    update_array[i].type = INSERT;
  }
  
  size_t update_array_index = 0;
  timer_start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < edges.size(); i++) {
    update_array[update_array_index].edge = edges[i];

    update_array_index++;
    if (update_array_index == update_array_size) {
      fout_part1.write_updates(update_array, update_array_size);
      update_array_index = 0;
    }
  }

  // Write any remaining edges to stream
  if (update_array_index > 0) {
		fout_part1.write_updates(update_array, update_array_index);
	}

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Writing time (sec): " << duration.count() << "\n";

  /*
  * Part 2: Read second half of binary stream, shuffle, then write
  */

  std::cout << "Processing second half:\n";

  // Initialize
  total_read_updates = 0;
  read_complete = false;

  timer_start = std::chrono::steady_clock::now();
  edges.resize(second_buffer_size, {0, 0});
  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Finished initializing buffer for edges: " << duration.count() << "s " <<  std::endl;

  // Reading edges
  std::cout << "  Reading edges...\n";
  timer_start = std::chrono::steady_clock::now();

  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    #pragma omp parallel for
    for (size_t i = 0; i < updates; i++) {
      if (read_complete) continue;

      GraphStreamUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);

      if (upd.type == BREAKPOINT) {
        read_complete = true;
      }
      else {
        edges[total_read_updates + i] = upd.edge;
      }
    }
    total_read_updates += updates;
  }
  // Decrement 1 for break point
  total_read_updates--;

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "    Total number of edges read: " << total_read_updates << "\n";
  std::cout << "  Reading time (sec): " << duration.count() << "\n";

  // Shuffling edges
  std::cout << "  Shuffling edges...\n";
  timer_start = std::chrono::steady_clock::now();
  std::shuffle(edges.begin(), edges.end(), rng);

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Shuffling time (sec): " << duration.count() << "\n";

  // Writing edges
  timer_start = std::chrono::steady_clock::now();
  std::cout << "  Writing edges back to stream...\n";
  BinaryFileStream fout_part2(output_name + "_part2", false);
  fout_part2.write_header(stream.vertices(), second_buffer_size);

  for (size_t i = 0; i < update_array_size; i++) {
    update_array[i].edge = {0, 0};
    update_array[i].type = INSERT;
  }
  
  update_array_index = 0;
  timer_start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < edges.size(); i++) {
    update_array[update_array_index].edge = edges[i];

    update_array_index++;
    if (update_array_index == update_array_size) {
      fout_part2.write_updates(update_array, update_array_size);
      update_array_index = 0;
    }
  }

  // Write any remaining edges to stream
  if (update_array_index > 0) {
		fout_part2.write_updates(update_array, update_array_index);
	}

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "  Writing time (sec): " << duration.count() << "\n";
}

void merge_partition(std::string stream_name) {
  std::string output_name = stream_name + "_shuffled";

  BinaryFileStream stream_part1(output_name + "_part1");
  BinaryFileStream stream_part2(output_name + "_part2");

  node_id_t num_vertices = stream_part1.vertices();
  size_t num_edges = stream_part1.edges() + stream_part2.edges();

  BinaryFileStream fout(output_name + "_merged", false);
  fout.write_header(num_vertices, num_edges);

  std::cout << "Merging streams:\n";

  // Perform read and write at the same time
  bool part1_read_complete = false, part2_read_complete = false;
  int update_size_factor = 10;
  size_t total_read_updates = 0;

  std::vector<GraphStreamUpdate> update_array_part1(update_array_size * update_size_factor);
  std::vector<GraphStreamUpdate> update_array_part2(update_array_size * update_size_factor);
  std::vector<GraphStreamUpdate> update_array_merged(update_array_size * update_size_factor);

  for (size_t i = 0; i < update_array_size * update_size_factor; i++) {
    update_array_merged[i].edge = {0, 0};
    update_array_merged[i].type = INSERT;
  }

  std::cout << "  Reading binary streams then merging...\n";
  auto timer_start = std::chrono::steady_clock::now();
  while (!part1_read_complete && !part2_read_complete) {
    size_t updates_part1 = stream_part1.get_update_buffer(update_array_part1.data(), (update_array_size * update_size_factor));
    size_t updates_part2 = stream_part2.get_update_buffer(update_array_part2.data(), (update_array_size * update_size_factor));

    #pragma omp parallel for
    for (size_t i = 0; i < updates_part1; i++) {
      if (part1_read_complete) continue;

      GraphStreamUpdate upd;
      upd.edge = update_array_part1[i].edge;
      upd.type = static_cast<UpdateType>(update_array_part1[i].type);

      if (upd.type == BREAKPOINT) {
        part1_read_complete = true;
      }
      else {
        update_array_merged[i] = upd;
      }
    }

    total_read_updates += updates_part1;

    if (part1_read_complete) {
      total_read_updates--; 
      fout.write_updates(update_array_merged.data(), updates_part1 - 1);
    }
    else {
      fout.write_updates(update_array_merged.data(), updates_part1);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < updates_part2; i++) {
      if (part2_read_complete) continue;

      GraphStreamUpdate upd;
      upd.edge = update_array_part2[i].edge;
      upd.type = static_cast<UpdateType>(update_array_part2[i].type);

      if (upd.type == BREAKPOINT) {
        part2_read_complete = true;
      }
      else {
        update_array_merged[i] = upd;
      }
    }

    total_read_updates += updates_part2;

    if (part2_read_complete) {
      total_read_updates--; 
      fout.write_updates(update_array_merged.data(), updates_part2 - 1);
    }
    else {
      fout.write_updates(update_array_merged.data(), updates_part2);
    }

  }

  // Read the remaining edges
  while (!part1_read_complete) {
    size_t updates_part1 = stream_part1.get_update_buffer(update_array_part1.data(), (update_array_size * update_size_factor));
    size_t rem_read_updates = 0;
    for (size_t i = 0; i < updates_part1; i++) {
      GraphStreamUpdate upd;
      upd.edge = update_array_part1[i].edge;
      upd.type = static_cast<UpdateType>(update_array_part1[i].type);

      if (upd.type == BREAKPOINT) {
        break;
      }
      else {
        update_array_merged[i] = upd;
        rem_read_updates++;
      }
    }
    fout.write_updates(update_array_merged.data(), rem_read_updates);
    total_read_updates += rem_read_updates;
  }

  while (!part2_read_complete) {
    size_t updates_part2 = stream_part2.get_update_buffer(update_array_part2.data(), (update_array_size * update_size_factor));
    size_t rem_read_updates = 0;
    for (size_t i = 0; i < updates_part2; i++) {
      GraphStreamUpdate upd;
      upd.edge = update_array_part2[i].edge;
      upd.type = static_cast<UpdateType>(update_array_part2[i].type);

      if (upd.type == BREAKPOINT) {
        break;
      }
      else {
        update_array_merged[i] = upd;
        rem_read_updates++;
      }
    }
    fout.write_updates(update_array_merged.data(), rem_read_updates);
    total_read_updates += rem_read_updates;
  }
  
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "    Total number of edges processed: " << total_read_updates << "\n";
  std::cout << "  Finished Merging (sec): " << duration.count() << "\n";
}

int main(int argc, char** argv) {

  if (argc != 3) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_file shuffle_in_ram" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string shuffle_in_ram = argv[2];
  if (shuffle_in_ram == "true") {
    std::cout << "Shuffling in RAM\n";
    inplace_shuffle(argv[1]);
  }
  else if (shuffle_in_ram == "false") {
    std::cout << "Partitioning the shuffle\n";
    partition_shuffle(argv[1]);
    merge_partition(argv[1]);
  }
  else {
    std::cout << "Incorrect value of 'shuffle_in_ram', must be 'true' or 'false'\n";
    exit(EXIT_FAILURE); 
  }
}