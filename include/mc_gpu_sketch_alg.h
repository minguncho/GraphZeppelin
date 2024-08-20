#pragma once

#include <map>
#include "mc_sketch_alg.h"
#include "cuda_kernel.cuh"
#include "edge_store.h"

struct CudaStream {
  cudaStream_t stream;
  int delta_applied;
  int src_vertex;
  int num_graphs;
};

struct SketchParams {
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;  
};

struct SketchSubgraph {
  std::atomic<size_t> num_updates;
  CudaUpdateParams* cudaUpdateParams = nullptr;
};

class MCGPUSketchAlg : public MCSketchAlg {
private:
  size_t sketchSeed;

  Bucket* delta_buckets;

  node_id_t num_nodes;
  int k;
  int sketches_factor;

  // Maximum number of subgraphs
  size_t num_subgraphs;

  // Current number of initialized sketch subgraphs. Starts at 0.
  std::atomic<node_id_t> cur_subgraphs;

  std::atomic<size_t> num_updates_seen;
  
  // Number of subgraphs in sketch representation
  int max_sketch_graphs; // Max. number of subgraphs that can be in sketch graphs

  // sketch subgraphs
  SketchSubgraph *subgraphs;

  // lossless edge storage
  EdgeStore edge_store;

  CudaKernel cudaKernel;

  // Variables from sketch
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads;
  int num_device_blocks;

  // Number of CPU's graph workers
  int num_host_threads;

  // Number of CPU threads that read edge stream
  int num_reader_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // Number of CUDA Streams per graph worker
  int stream_multiplier = 4;

  // Vector for storing information for each CUDA Stream
  std::vector<CudaStream> streams;

  // helper functions for apply_update_batch()
  size_t get_and_apply_finished_stream(int thr_id);
  void complete_update_batch(int thr_id, const TaggedUpdateBatch &updates);

  std::mutex sketch_creation_lock;
public:
 MCGPUSketchAlg(node_id_t num_vertices, int num_threads, int _num_reader_threads, size_t seed,
                SketchParams sketchParams, int _num_subgraphs, int _max_sketch_graphs, int _k,
                size_t _sketch_bytes, int _initial_sketch_graphs, CCAlgConfiguration config)
     : MCSketchAlg(num_vertices, seed, _max_sketch_graphs, config),
       edge_store(seed, num_vertices, _sketch_bytes, _num_subgraphs, _initial_sketch_graphs) {
    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_updates_seen = 0;
    sketchSeed = seed;
    num_nodes = num_vertices;
    k = _k;
    sketches_factor = config.get_sketches_factor();
    num_host_threads = num_threads;
    num_reader_threads = _num_reader_threads;

    num_device_threads = 1024;
    num_device_blocks = k;  // Change this value based on dataset <-- Come up with formula to compute
                           // this automatically

    if (_max_sketch_graphs < _initial_sketch_graphs) {
      std::cerr << "ERROR: Cannot have initial sketch graphs > max sketch graphs" << std::endl;
      exit(EXIT_FAILURE);
    }
    num_subgraphs = _num_subgraphs;
    cur_subgraphs = _initial_sketch_graphs;
    max_sketch_graphs = _max_sketch_graphs;
    subgraphs = new SketchSubgraph[max_sketch_graphs];

    // Extract sketchParams variables
    num_samples = sketchParams.num_samples;
    num_columns = sketchParams.num_columns;
    bkt_per_col = sketchParams.bkt_per_col;
    num_buckets = sketchParams.num_buckets;

    std::cout << "num_samples: " << num_samples << "\n";
    std::cout << "num_buckets: " << num_buckets << "\n";
    std::cout << "num_columns: " << num_columns << "\n";
    std::cout << "bkt_per_col: " << bkt_per_col << "\n";

    // Initialize delta_buckets
    delta_buckets = new Bucket[num_buckets * num_host_threads];

    // Create a bigger batch size to apply edge updates when subgraph is turning into sketch
    // representation
    batch_size = get_desired_updates_per_batch();

    if (max_sketch_graphs > 0) {  // If max_sketch_graphs is 0, there will never be any sketch graphs
      int device_id = cudaGetDevice(&device_id);
      int device_count = 0;
      cudaGetDeviceCount(&device_count);
      std::cout << "CUDA Device Count: " << device_count << "\n";
      std::cout << "CUDA Device ID: " << device_id << "\n";

      // Calculate the num_buckets assigned to the last thread block
      // size_t num_last_tb_buckets =
      //     (subgraphs[0].cudaUpdateParams->num_tb_columns[num_device_blocks - 1] * bkt_per_col) + 1;

      // // Set maxBytes for GPU kernel's shared memory
      // size_t maxBytes =
      //     (num_last_tb_buckets * sizeof(vec_t_cu)) + (num_last_tb_buckets * sizeof(vec_hash_t));
      // cudaKernel.updateSharedMemory(maxBytes);
      // std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";
    }

    // Initialize Sketch Graphs
    for (int i = 0; i < cur_subgraphs; i++) {
      create_sketch_graph(i);

      CudaUpdateParams* params;
      // TODO: Is this malloc necessary?
      gpuErrchk(cudaMallocManaged(&params, sizeof(CudaUpdateParams)));
      params = new CudaUpdateParams(
         num_nodes, num_samples, num_buckets, num_columns, bkt_per_col, num_host_threads,
         num_reader_threads, batch_size, stream_multiplier, num_device_blocks, k);
      subgraphs[i].num_updates = 0;
      subgraphs[i].cudaUpdateParams = params;
    }

    // Initialize CUDA Streams
    for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
      cudaStream_t stream;

      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      streams.push_back({stream, 1, -1, -1});
    }

    std::cout << "Finished MCGPUSketchAlg's Initialization" << std::endl;
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "MCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  }

  ~MCGPUSketchAlg() {
    for (size_t i = 0; i < cur_subgraphs; i++)
      delete subgraphs[i].cudaUpdateParams;
    delete[] subgraphs;
    delete[] delta_buckets;
  }

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  // Update with the delta sketches that haven't been applied yet.
  void apply_flush_updates();

  void print_subgraph_edges() {
    std::cout << "Number of inserted updates for each subgraph:\n";
    for (int graph_id = 0; graph_id < cur_subgraphs.load(); graph_id++) {
    std::cout << "  Sub-Graph " << graph_id << "(Sketch): " << subgraphs[graph_id].num_updates
                << std::endl;
    }
    std::cout << "  Adjacency list:      " << edge_store.get_num_edges() << std::endl;
  }

  std::vector<Edge> get_adjlist_spanning_forests();
  int get_num_sketch_graphs() { return cur_subgraphs; }

  size_t get_num_adjlist_edges() { return edge_store.get_num_edges(); }
};
