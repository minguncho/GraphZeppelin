#pragma once

#include <cmath>
#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"
#include "cuda_stream.h"

struct SketchParams {
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;
};

class CCGPUSketchAlg : public CCSketchAlg{
private:
  CudaUpdateParams* cudaUpdateParams;
  size_t sketchSeed;

  CudaKernel cudaKernel;

  // Variables from sketch
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads = 1024;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  int batch_size;

  // Number of batches in a buffer to accept until GPU kernel launch
  int num_batch_per_buffer = 1080;
  CudaStream** cudaStreams;

public:
  CCGPUSketchAlg(node_id_t num_vertices, size_t num_updates, int num_threads, Bucket* buckets, size_t seed, SketchParams sketchParams, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(num_vertices, seed, buckets, config){ 

    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_host_threads = num_threads;
    sketchSeed = seed;

    // Get variables from sketch
    num_samples = sketchParams.num_samples;
    num_columns = sketchParams.num_columns;
    bkt_per_col = sketchParams.bkt_per_col;
    num_buckets = sketchParams.num_buckets;

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

    // Create cudaUpdateParams
    gpuErrchk(cudaMallocManaged(&cudaUpdateParams, sizeof(CudaUpdateParams)));
    cudaUpdateParams = new CudaUpdateParams(num_vertices, num_updates, buckets, num_samples, num_buckets, num_columns, bkt_per_col, num_threads, batch_size);

    // Set maxBytes for GPU kernel's shared memory
    size_t maxBytes = (num_buckets * sizeof(vec_t_cu)) + (num_buckets * sizeof(vec_hash_t));
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Initialize CUDA Streams
    cudaStreams = new CudaStream*[num_host_threads];
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      cudaStreams[thr_id] = new CudaStream(num_device_threads, num_batch_per_buffer, batch_size, cudaUpdateParams, sketchSeed);
    }

    std::cout << "Num batches per buffer: " << num_batch_per_buffer << "\n";

    // Prefetch sketches to GPU
    gpuErrchk(cudaMemPrefetchAsync(buckets, num_vertices * sketchParams.num_buckets * sizeof(Bucket), device_id));

    std::cout << "Finished CCGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "CCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  void flush_buffers();

};