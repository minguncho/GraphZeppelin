#include "cc_gpu_sketch_alg.h"

#include <iostream>
#include <thread>
#include <vector>

void CCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();

  cudaStreams[thr_id]->process_batch(src_vertex, dst_vertices);
};

void CCGPUSketchAlg::flush_buffers() {

  auto flush_start = std::chrono::steady_clock::now();
  // Original Method
  for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
    cudaStreams[thr_id]->flush_buffers();
  }

  cudaDeviceSynchronize();
  std::chrono::duration<double> flush_time = std::chrono::steady_clock::now() - flush_start;

  int longest_thr_id = 0;
  double longest_process_time = 0;
  
  for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
    std::cout << "Thread # " << thr_id << ": " << cudaStreams[thr_id]->process_time.count() << "\n";
    std::cout << "  Buffer Time: " << cudaStreams[thr_id]->buffer_time.count() << "\n";
    std::cout << "    Edge Convert Time: " << cudaStreams[thr_id]->edge_convert_time.count() << "\n";
    std::cout << "  CUDA Stream Wait Time: " << cudaStreams[thr_id]->wait_time.count() << "\n";

    if (cudaStreams[thr_id]->process_time.count() > longest_process_time) {
      longest_process_time = cudaStreams[thr_id]->process_time.count();
      longest_thr_id = thr_id;
    }
  }
  std::cout << "\n";
  std::cout << "Longest Thread # " << longest_thr_id << ": " << cudaStreams[longest_thr_id]->process_time.count() << "\n";
  std::cout << "  Buffer Time: " << cudaStreams[longest_thr_id]->buffer_time.count() << "\n";
  std::cout << "    Edge Convert Time: " << cudaStreams[longest_thr_id]->edge_convert_time.count() << "\n";
  std::cout << "  CUDA Stream Wait Time: " << cudaStreams[longest_thr_id]->wait_time.count() << "\n"; 
  std::cout << "GPU Flushing buffers time: " << flush_time.count() << "\n";
}