#pragma once
#include <chrono>
#include <thread>
#include <vector>

#include <sketch.h>

struct SketchParams {
  // Sketch related variables
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;
  size_t seed;
};

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes start_density max_density density_inc num_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  node_id_t num_nodes = std::atoi(argv[1]);
  double start_density = std::stod(argv[2]);
  double max_density = std::stod(argv[3]);
  double density_inc = std::stod(argv[4]);
  size_t num_complete_edges = ((num_nodes * (num_nodes - 1)) / 2);
  int num_threads = std::atoi(argv[5]);

  std::cout << "Max Density: " << max_density * 100 << "%\n";

  size_t sketchSeed = get_seed();

  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  std::cout << "-----Sketch Information-----\n";
  std::cout << "num_nodes: " << num_nodes << "\n";
  std::cout << "num_complete_edges: " << num_complete_edges << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "\n";

  size_t num_updates_per_batch = (sketchParams.num_buckets * sizeof(Bucket)) / sizeof(node_id_t);
  size_t num_max_batches = std::ceil(((double)num_complete_edges * 2) / num_updates_per_batch);

  std::cout << "Max Number of Batches: " << num_max_batches << "\n";
  std::cout << "Batch Size: " << num_updates_per_batch << "\n";

  Sketch **delta_sketches = new Sketch *[num_threads];
  for (size_t thr_id = 0; thr_id < num_threads; thr_id++) {
    delta_sketches[thr_id] = new Sketch(Sketch::calc_vector_length(num_nodes), sketchSeed, Sketch::calc_cc_samples(num_nodes, 1));
  }

  for (double density = start_density; density < (max_density + 0.000001); density += density_inc) {
    int num_batches = density * num_max_batches;

    std::cout << "Density: " << density * 100 << "%\n";
    std::cout << "  Number of batches: " << num_batches << "\n";
    std::cout << "  Number of updates: " << num_updates_per_batch * num_batches << "\n";

    if (num_batches == 0) {
      std::cout << "  Current Density too low, skipping\n";
      continue;
    }

    auto sketch_update_start = std::chrono::steady_clock::now();

    auto task = [&](int thr_id) {
      for (int batch_id = thr_id; batch_id < num_batches; batch_id += num_threads) {
        // Reset delta sketch
        delta_sketches[thr_id]->zero_contents();

        node_id_t src_vertex = batch_id / num_nodes;

        for (int update_id = 0; update_id < num_updates_per_batch; update_id++) {
          delta_sketches[thr_id]->update(static_cast<vec_t>(concat_pairing_fn(src_vertex, update_id)));
        }
      }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; i++) threads.emplace_back(task, i);

    // wait for threads to finish
    for (size_t i = 0; i < num_threads; i++) threads[i].join();

    std::chrono::duration<double> sketch_update_duration = std::chrono::steady_clock::now() - sketch_update_start;
    std::cout << "Total insertion time(sec):    " << sketch_update_duration.count() << std::endl;
    std::cout << "Updates per second:           " << ((num_updates_per_batch * num_batches) / 2) / sketch_update_duration.count() << std::endl;

  }
  
}
