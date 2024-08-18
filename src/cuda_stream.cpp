#include "cuda_stream.h"
#include "util.h"

// Constructor
CudaStream::CudaStream(int num_device_threads, int num_batch_per_buffer, size_t batch_size, SketchParams sketchParams)
    : num_device_threads(num_device_threads), num_batch_per_buffer(num_batch_per_buffer), batch_size(batch_size), sketchParams(sketchParams) {

  // Initialize CudaStream
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Allocate buffers for batches
  gpuErrchk(cudaMallocHost(&h_edgeUpdates, 2 * num_batch_per_buffer * batch_size * sizeof(node_id_t)));
  gpuErrchk(cudaMalloc(&d_edgeUpdates, num_batch_per_buffer * batch_size * sizeof(node_id_t)));

  // Allocate buffers for batch information
  gpuErrchk(cudaMallocHost(&h_update_sizes, 2 * num_batch_per_buffer * batch_size * sizeof(vec_t)));
  gpuErrchk(cudaMallocHost(&h_update_src, 2 * num_batch_per_buffer * batch_size * sizeof(node_id_t)));
  gpuErrchk(cudaMallocHost(&h_update_start_index, 2 * num_batch_per_buffer * batch_size * sizeof(vec_t)));
  gpuErrchk(cudaMalloc(&d_update_sizes, num_batch_per_buffer * batch_size * sizeof(vec_t)));
  gpuErrchk(cudaMalloc(&d_update_src, num_batch_per_buffer * batch_size * sizeof(node_id_t)));
  gpuErrchk(cudaMalloc(&d_update_start_index, num_batch_per_buffer * batch_size * sizeof(vec_t)));

  // Initialize buffers
  for (int i = 0; i < 2 * num_batch_per_buffer * batch_size; i++) {
    h_edgeUpdates[i] = 0;
    h_update_sizes[i] = 0;
    h_update_src[i] = 0;
    h_update_start_index[i] = 0;
  }

  batch_limit = num_batch_per_buffer * batch_size;
}

void CudaStream::process_batch(node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices) {
  auto process_start = std::chrono::steady_clock::now();
  size_t start_index = buffer_id * num_batch_per_buffer * batch_size;

  if (batch_offset + dst_vertices.size() > batch_limit) { // Buffer will go over with new batch, start GPU
    auto start = std::chrono::steady_clock::now();
    gpuErrchk(cudaStreamSynchronize(stream)); // Make sure CUDA Stream has finished working on previous buffer
    wait_time += std::chrono::steady_clock::now() - start;

    // Transfer buffers
    gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, &h_edgeUpdates[start_index], (batch_offset - start_index) * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_sizes, &h_update_sizes[start_index], batch_count * sizeof(vec_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_src, &h_update_src[start_index], batch_count * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_start_index, &h_update_start_index[start_index], batch_count * sizeof(vec_t), cudaMemcpyHostToDevice, stream));

    // Prefetch sketches to GPU
    /*auto prefetch_start = std::chrono::steady_clock::now();
    for (int batch_id = 0; batch_id < batch_count; batch_id++) {
      gpuErrchk(cudaMemPrefetchAsync(&(sketchParams.buckets[h_update_src[batch_id] * sketchParams.num_buckets]), sketchParams.num_buckets * sizeof(Bucket), 0, stream));
    }
    prefetch_time += std::chrono::steady_clock::now() - prefetch_start;*/

    // Launch GPU kernel
    cudaKernel.sketchUpdate(num_device_threads, batch_count, stream, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);

    // Reset variables
    batch_count = 0;
    if (buffer_id == 0) {
      buffer_id = 1;
      batch_offset = num_batch_per_buffer * batch_size;
      batch_limit = 2 * num_batch_per_buffer * batch_size;
    }
    else { // Buffer id = 1
      buffer_id = 0;
      batch_offset = 0;
      batch_limit = num_batch_per_buffer * batch_size;
    }

    // Recalculate start_index
    start_index = buffer_id * num_batch_per_buffer * batch_size;
  }

  auto edge_fill_start = std::chrono::steady_clock::now();
  int count = 0;
  for (vec_t i = batch_offset; i < batch_offset + dst_vertices.size(); i++) {
    h_edgeUpdates[i] = dst_vertices[count];
    count++;
  }
  edge_fill_time += std::chrono::steady_clock::now() - edge_fill_start;

  h_update_sizes[start_index + batch_count] = dst_vertices.size();
  h_update_src[start_index + batch_count] = src_vertex;
  h_update_start_index[start_index + batch_count] = batch_offset - start_index;

  batch_offset += dst_vertices.size();
  batch_count++;

  process_time += std::chrono::steady_clock::now() - process_start;
}

void CudaStream::flush_buffers() {
  if (batch_count == 0) return;
  int num_batches_left = batch_count;
  int start_index = buffer_id * num_batch_per_buffer * batch_size;

  //cudaStreamSynchronize(stream);

  //std::cout << "Within Stream, batches left: " << batch_count << ", offset: " << (batch_offset - start_index) << "\n";

  // Transfer buffers
  gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, &h_edgeUpdates[start_index], (batch_offset - start_index) * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_update_sizes, &h_update_sizes[start_index], num_batches_left * sizeof(vec_t), cudaMemcpyHostToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_update_src, &h_update_src[start_index], num_batches_left * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_update_start_index, &h_update_start_index[start_index], num_batches_left * sizeof(vec_t), cudaMemcpyHostToDevice, stream));

  cudaKernel.sketchUpdate(num_device_threads, num_batches_left, stream, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);
}
