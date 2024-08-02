#include "cuda_kernel.cuh"
#include <chrono>

class CudaStream {
private:
  CudaUpdateParams* cudaUpdateParams;
  cudaStream_t stream;

  CudaKernel cudaKernel;

  node_id_t *h_edgeUpdates, *d_edgeUpdates;
  vec_t *h_update_sizes, *d_update_sizes, *h_update_start_index, *d_update_start_index;
  node_id_t *h_update_src, *d_update_src;

  int num_batch_per_buffer;
  
  int buffer_id = 0;
  size_t batch_offset = 0;
  int batch_count = 0;
  size_t batch_size;

  size_t sketchSeed;

  int num_device_threads;

public:
  // Constructor
  CudaStream(int num_device_threads, int num_batch_per_buffer, size_t batch_sizes, CudaUpdateParams* cudaUpdateParams, size_t sketchSeed);

  void process_batch(node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices);
  void flush_buffers();

  std::chrono::duration<double> wait_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> process_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> buffer_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> edge_convert_time = std::chrono::nanoseconds::zero();
};