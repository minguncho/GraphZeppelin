#include "sk_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

void SKGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();
  // Get offset
  size_t offset = edgeUpdate_offset.fetch_add(dst_vertices.size());

  // Fill in buffer
  size_t index = 0;
  for (auto dst : dst_vertices) {
    h_edgeUpdates[offset + index] = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst));
    index++;
  }

  size_t batch_id = batch_count.fetch_add(1);
  std::lock_guard<std::mutex> lk(batch_mutex);
  batch_sizes.insert({batch_id, dst_vertices.size()});
  batch_src.insert({batch_id, src_vertex});
  batch_start_index.insert({batch_id, offset});
};

void SKGPUSketchAlg::launch_gpu_kernel() {
  // Declare GPU block count and size
  num_device_threads = 1024;
  std::cout << "Num GPU threads per block: " << num_device_threads << "\n";
  std::cout << "Number of batches: " << batch_count << "\n";

  std::cout << "Preparing update buffers for GPU...\n";
  gpuErrchk(cudaMallocHost(&h_update_sizes, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMallocHost(&h_update_src, batch_count * sizeof(node_id_t)));
  gpuErrchk(cudaMallocHost(&h_update_start_index, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMalloc(&d_update_sizes, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMalloc(&d_update_src, batch_count * sizeof(node_id_t)));
  gpuErrchk(cudaMalloc(&d_update_start_index, batch_count * sizeof(vec_t)));
  // Fill in update_sizes and update_src
  for (auto it = batch_sizes.begin(); it != batch_sizes.end(); it++) {
    h_update_sizes[it->first] = it->second;
    h_update_src[it->first] = batch_src[it->first]; 
    h_update_start_index[it->first] = batch_start_index[it->first];
  }
  
  // Transfer buffers to GPU
  gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, 2 * num_updates * sizeof(vec_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_update_sizes, h_update_sizes, batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_update_src, h_update_src, batch_count * sizeof(node_id_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_update_start_index, h_update_start_index, batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));

  int min_device_block = (batch_count / num_device_threads) + 1;

  /*for (int i = 1; i <= 100; i++) {
    memset(cudaUpdateParams->buckets, 0, num_nodes * num_buckets * sizeof(Bucket));
    // Prefetch sketches to GPU
    gpuErrchk(cudaMemPrefetchAsync(cudaUpdateParams->buckets, num_nodes * num_buckets * sizeof(Bucket), device_id));

    //num_device_blocks = i * min_device_block * 10;
    num_device_blocks = i * 10;
    // Launch GPU kernel
    std::cout << "Launching GPU Kernel... Iteration: " << i << " num_device_blocks = " << num_device_blocks;
    auto kernel_start = std::chrono::steady_clock::now();
    cudaKernel.single_sketchUpdate(num_device_threads, num_device_blocks, num_thread_blocks_per_batch, batch_count, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, cudaUpdateParams, sketchSeed);
    cudaDeviceSynchronize();
    auto kernel_end = std::chrono::steady_clock::now();
    
    std::cout << "  GPU Kernel Finished.\n";
    std::chrono::duration<double> kernel_time = kernel_end - kernel_start;
    std::cout << "    Elapsed Time: " << kernel_time.count() << "\n";
    std::cout << "    Throughput: " << num_updates / kernel_time.count() << "\n";
  }*/


  // Launch GPU kernel
  num_device_blocks = batch_count;
  std::cout << "Num GPU thread blocks: " << num_device_blocks << "\n";
  std::cout << "Launching GPU Kernel...\n";
  auto kernel_start = std::chrono::steady_clock::now();
  cudaKernel.single_sketchUpdate(num_device_threads, num_device_blocks, batch_count, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, cudaUpdateParams, sketchSeed);
  cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();
  
  std::cout << "  GPU Kernel Finished.\n";
  std::chrono::duration<double> kernel_time = kernel_end - kernel_start;
  std::cout << "    Elapsed Time: " << kernel_time.count() << "\n";
  std::cout << "    Throughput: " << num_updates / kernel_time.count() << "\n";

  // Prefecth buffers back to CPU
  gpuErrchk(cudaMemPrefetchAsync(cudaUpdateParams->buckets, num_nodes * num_buckets * sizeof(Bucket), cudaCpuDeviceId));
}