#include "bucket.h"
#include <cuda_xxhash64.cuh>
#include "../src/cuda_kernel.cu"
#include "util.h"
#include <gtest/gtest.h>

#include <iostream>
#include <chrono>

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

__global__ void gpuKernel(vec_t* updates, vec_t* gpu_hash_values, int N, uint64_t seed, size_t num_columns, size_t bkt_per_col) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(tid < N) {
    int column_id = tid / num_columns;
    gpu_hash_values[tid] = bucket_get_index_depth(updates[tid], seed + (column_id * 5), bkt_per_col);
  }
}

TEST(CUDAXXHashTest, TestCheckHashValues) {
  vec_t* updates;

  uint64_t seed = get_seed();
  std::cout << "Seed: " << seed << "\n";

  size_t num_columns = 1000;
  size_t bkt_per_col = 100;
  size_t N = num_columns * bkt_per_col;

  // CPU 
  vec_t* cpu_hash_values = new vec_t[N];

  // GPU
  vec_t* gpu_hash_values;

  cudaMallocManaged(&updates, N * sizeof(vec_t));
  cudaMallocManaged(&gpu_hash_values, N * sizeof(vec_t));

  // Initialization
  for (node_id_t i = 0; i < N; i++) {
    updates[i] = static_cast<vec_t>(concat_pairing_fn(i, i));
    cpu_hash_values[i] = 0;
    gpu_hash_values[i] = 0;
  }

  // Run CPU version
  for (node_id_t i = 0; i < N; i++) {
    //cpu_hash_values[i] = Bucket_Boruvka::get_index_hash(updates[i], seed);
    int column_id = i / num_columns;
    cpu_hash_values[i] = Bucket_Boruvka::get_index_depth(updates[i], seed + (column_id * 5), bkt_per_col);
  }

  int num_threads = 1024;
  int num_blocks = (N + num_threads - 1) / num_threads;

  // Run GPU Kernel
  gpuKernel<<<num_blocks,num_threads>>>(updates, gpu_hash_values, N, seed, num_columns, bkt_per_col);
  cudaDeviceSynchronize();

  // Validate values
  for (node_id_t i = 0; i < N; i++) {
    ASSERT_EQ(cpu_hash_values[i], gpu_hash_values[i]);
  }
}