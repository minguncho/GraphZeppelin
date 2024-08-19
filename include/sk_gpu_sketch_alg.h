#pragma once

#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"

class SKGPUSketchAlg : public CCSketchAlg{
private:
  SketchParams sketchParams;
  size_t maxBytes;

  CudaKernel cudaKernel;

  node_id_t num_nodes;
  size_t num_updates;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads;
  int num_device_blocks;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // List of edge ids that thread will be responsble for updating
  node_id_t *h_edgeUpdates, *d_edgeUpdates;

  vec_t *h_update_sizes, *d_update_sizes, *h_update_start_index, *d_update_start_index;
  node_id_t *h_update_src, *d_update_src;

  std::map<uint64_t, uint64_t> batch_sizes;
  std::map<uint64_t, uint64_t> batch_src;
  std::map<uint64_t, uint64_t> batch_start_index;
  std::mutex batch_mutex;

  // Atomic variables
  std::atomic<uint64_t> edgeUpdate_offset;
  std::atomic<uint64_t> batch_count;

public:
  SKGPUSketchAlg(node_id_t _num_nodes, size_t _num_updates, int num_threads, SketchParams _sketchParams, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(_num_nodes, _sketchParams.cudaUVM_enabled, _sketchParams.seed, _sketchParams.cudaUVM_buckets, config){ 

    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_nodes = _num_nodes;
    num_updates = _num_updates;

    edgeUpdate_offset = 0;
    batch_count = 0;
    
    num_host_threads = num_threads;
    sketchParams = _sketchParams;

    batch_size = get_desired_updates_per_batch();
    std::cout << "Batch Size: " << batch_size << "\n";

    int device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";
    std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n"; 

    // Set maxBytes for GPU kernel's shared memory
    maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t));
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Allocate memory for buffer that stores edge updates
    gpuErrchk(cudaMallocHost(&h_edgeUpdates, 2 * num_updates * sizeof(node_id_t)));
    gpuErrchk(cudaMalloc(&d_edgeUpdates, 2 * num_updates * sizeof(node_id_t)));

    // Initialize buffer with 0
    memset(h_edgeUpdates, 0, 2 * num_updates * sizeof(node_id_t));

    // Prefetch sketches to GPU
    gpuErrchk(cudaMemPrefetchAsync(sketchParams.cudaUVM_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket), device_id));
    
    std::cout << "Bucket Memory Bytes: " << (double)(num_nodes * sketchParams.num_buckets * sizeof(Bucket)) / 1000000000 << "GB\n";

    std::cout << "Finished SKGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "SKGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };
  
  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  void launch_gpu_kernel();
  void launch_multiple_gpu_kernels();
  uint64_t get_batch_count() { 
    uint64_t temp = batch_count;
    return temp; }
};