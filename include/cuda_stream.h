#pragma once
#include "cuda_kernel.cuh"
#include <sys/resource.h> // for rusage
#include <chrono>

static double test_get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}


template <class Alg>
class CudaStream {
private:
  Alg *sketching_alg;
  int graph_id;
  node_id_t num_nodes;
  SketchParams sketchParams;
  cudaStream_t stream;

  CudaKernel cudaKernel;

  node_id_t *h_edgeUpdates, *d_edgeUpdates;
  vec_t *h_update_sizes, *d_update_sizes, *h_update_start_index, *d_update_start_index;
  node_id_t *h_update_src, *d_update_src;

  Bucket *h_buckets, *d_buckets;

  int num_batch_per_buffer;
  
  int buffer_id = 0;
  size_t batch_offset = 0;
  size_t batch_size;
  size_t batch_limit;
  size_t batch_count = 0;

  size_t sketchSeed;

  int num_device_threads;
  bool first_buffer = true;

public:
  // Constructor
  CudaStream(Alg *sketching_alg, int graph_id, node_id_t num_nodes, int num_device_threads, int num_batch_per_buffer, SketchParams _sketchParams)
    : sketching_alg(sketching_alg), graph_id(graph_id), num_nodes(num_nodes), num_device_threads(num_device_threads), num_batch_per_buffer(num_batch_per_buffer), sketchParams(_sketchParams) {

    // Initialize CudaStream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    batch_size = sketching_alg->get_desired_updates_per_batch();

    if (sketchParams.cudaUVM_enabled) {
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
    else {
      // To save memory space, let each batch size = sketch size which is true while GTS is being filled with edges. 
      // However during GTS' flushing process, buffer can be sent to GPU without being full.  

      // Allocate buffers for batches
      gpuErrchk(cudaMallocHost(&h_edgeUpdates, 2 * num_batch_per_buffer * batch_size * sizeof(node_id_t)));
      gpuErrchk(cudaMalloc(&d_edgeUpdates, num_batch_per_buffer * batch_size * sizeof(node_id_t)));

      // Allocate buffers for batch information
      gpuErrchk(cudaMallocHost(&h_update_sizes, 2 * num_batch_per_buffer * sizeof(vec_t)));
      gpuErrchk(cudaMallocHost(&h_update_src, 2 * num_batch_per_buffer * sizeof(node_id_t)));
      gpuErrchk(cudaMallocHost(&h_update_start_index, 2 * num_batch_per_buffer * sizeof(vec_t)));
      gpuErrchk(cudaMalloc(&d_update_sizes, num_batch_per_buffer * sizeof(vec_t)));
      gpuErrchk(cudaMalloc(&d_update_src, num_batch_per_buffer * sizeof(node_id_t)));
      gpuErrchk(cudaMalloc(&d_update_start_index, num_batch_per_buffer * sizeof(vec_t)));

      gpuErrchk(cudaMallocHost(&h_buckets, sketchParams.num_buckets * num_batch_per_buffer * sizeof(Bucket)));
      gpuErrchk(cudaMalloc(&sketchParams.d_buckets, sketchParams.num_buckets * num_batch_per_buffer * sizeof(Bucket)));

      // Initialize buffers
      for (int i = 0; i < 2 * num_batch_per_buffer * batch_size; i++) {
        h_edgeUpdates[i] = 0;
      }

      for (int i = 0; i < 2 * num_batch_per_buffer; i++) {
        h_update_sizes[i] = 0;
        h_update_src[i] = 0;
        h_update_start_index[i] = 0;
      }

      batch_limit = num_batch_per_buffer;
    }

  }

  void process_batch_UVM(node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices) {
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
      buffer_id = buffer_id ^ 1;
      batch_offset = buffer_id * num_batch_per_buffer * batch_size;
      batch_limit = (buffer_id + 1) * num_batch_per_buffer * batch_size;
      
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

  void process_batch_default(node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices) {
    auto process_start = std::chrono::steady_clock::now();
    size_t start_index = buffer_id * num_batch_per_buffer * batch_size;

    auto edge_fill_start = std::chrono::steady_clock::now();
    int count = 0;
    for (vec_t i = batch_offset; i < batch_offset + dst_vertices.size(); i++) {
      h_edgeUpdates[i] = dst_vertices[count];
      count++;
    }
    edge_fill_time += std::chrono::steady_clock::now() - edge_fill_start;

    int start_batch_id = buffer_id * num_batch_per_buffer;
    h_update_sizes[start_batch_id + batch_count] = dst_vertices.size();
    h_update_src[start_batch_id + batch_count] = src_vertex;
    h_update_start_index[start_batch_id + batch_count] = batch_offset - start_index;

    batch_offset += dst_vertices.size();
    batch_count++;

    if (batch_count == batch_limit) { // Buffer will go over with new batch, start GPU
      auto start = std::chrono::steady_clock::now();
      gpuErrchk(cudaStreamSynchronize(stream)); // Make sure CUDA Stream has finished working on previous buffer
      wait_time += std::chrono::steady_clock::now() - start;

      // Apply delta sketches
      if (!first_buffer) { // No delta sketch to apply when handling the very first buffer
        auto apply_delta_start = std::chrono::steady_clock::now();
        int prev_buffer_id = buffer_id ^ 1;
        size_t prev_batch_id = prev_buffer_id * num_batch_per_buffer;

        for (int batch_id = 0; batch_id < batch_limit; batch_id++) {
          node_id_t prev_src = h_update_src[prev_batch_id + batch_id];
          prev_src = (graph_id * num_nodes) + prev_src;
          sketching_alg->apply_raw_buckets_update(prev_src, &h_buckets[batch_id * sketchParams.num_buckets]);
        }
        apply_delta_time += std::chrono::steady_clock::now() - apply_delta_start;
      }

      // Transfer buffers
      gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, &h_edgeUpdates[start_index], (batch_offset - start_index) * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_update_sizes, &h_update_sizes[start_batch_id], batch_count * sizeof(vec_t), cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_update_src, &h_update_src[start_batch_id], batch_count * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_update_start_index, &h_update_start_index[start_batch_id], batch_count * sizeof(vec_t), cudaMemcpyHostToDevice, stream));

      // Launch GPU kernel
      cudaKernel.sketchUpdate(num_device_threads, batch_count, stream, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);

      // Queue up delta sketches transfer back to CPU
      gpuErrchk(cudaMemcpyAsync(h_buckets, sketchParams.d_buckets, sketchParams.num_buckets * num_batch_per_buffer * sizeof(Bucket), cudaMemcpyDeviceToHost, stream));

      // Reset variables
      batch_count = 0;
      buffer_id = buffer_id ^ 1;
      batch_offset = buffer_id * num_batch_per_buffer * batch_size;
      first_buffer = false;
    }

    process_time += std::chrono::steady_clock::now() - process_start;
  }

  void process_batch(node_id_t src_vertex, const std::vector<node_id_t> &dst_vertices) {
    if (sketchParams.cudaUVM_enabled) {
      process_batch_UVM(src_vertex, dst_vertices);
    }
    else {
      process_batch_default(src_vertex, dst_vertices);
    }
  }

  void flush_buffers_UVM() {
    if (batch_count == 0) return;
    int num_batches_left = batch_count;
    int start_index = buffer_id * num_batch_per_buffer * batch_size;

    // Transfer buffers
    gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, &h_edgeUpdates[start_index], (batch_offset - start_index) * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_sizes, &h_update_sizes[start_index], num_batches_left * sizeof(vec_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_src, &h_update_src[start_index], num_batches_left * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_start_index, &h_update_start_index[start_index], num_batches_left * sizeof(vec_t), cudaMemcpyHostToDevice, stream));

    cudaKernel.sketchUpdate(num_device_threads, num_batches_left, stream, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);
  }

  void flush_buffers_default() {
    gpuErrchk(cudaStreamSynchronize(stream));

    // Apply delta sketches
    int prev_buffer_id = buffer_id ^ 1;
    size_t prev_batch_id = prev_buffer_id * num_batch_per_buffer;

    for (int batch_id = 0; batch_id < batch_limit; batch_id++) {
      node_id_t prev_src = h_update_src[prev_batch_id + batch_id];
      prev_src = (graph_id * num_nodes) + prev_src;
      sketching_alg->apply_raw_buckets_update(prev_src, &h_buckets[batch_id * sketchParams.num_buckets]);
    }

    if (batch_count == 0) return;
    int num_batches_left = batch_count;
    int start_index = buffer_id * num_batch_per_buffer * batch_size;
    int start_batch_id = buffer_id * num_batch_per_buffer;

    // Transfer remaining buffers
    gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, &h_edgeUpdates[start_index], (batch_offset - start_index) * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_sizes, &h_update_sizes[start_batch_id], num_batches_left * sizeof(vec_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_src, &h_update_src[start_batch_id], num_batches_left * sizeof(node_id_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_update_start_index, &h_update_start_index[start_batch_id], num_batches_left * sizeof(vec_t), cudaMemcpyHostToDevice, stream));

    cudaKernel.sketchUpdate(num_device_threads, num_batches_left, stream, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);

    gpuErrchk(cudaMemcpyAsync(h_buckets, sketchParams.d_buckets, sketchParams.num_buckets * num_batches_left * sizeof(Bucket), cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    // Apply delta sketches
    for (int batch_id = 0; batch_id < num_batches_left; batch_id++) {
      node_id_t src = h_update_src[start_batch_id + batch_id];
      src = (graph_id * num_nodes) + src;
      sketching_alg->apply_raw_buckets_update(src, &h_buckets[batch_id * sketchParams.num_buckets]);
    }
  }
  
  void flush_buffers() {
    if (sketchParams.cudaUVM_enabled) {
      flush_buffers_UVM();
    }
    else {
      flush_buffers_default();
    }
  }

  std::chrono::duration<double> wait_time = std::chrono::nanoseconds::zero(); // Cumulative wait time for prev buffer to finish
  std::chrono::duration<double> process_time = std::chrono::nanoseconds::zero(); // Cumulative time to process a batch
  std::chrono::duration<double> edge_fill_time = std::chrono::nanoseconds::zero(); // Cumulative time to fill up a buffer with batch of edge updates

  std::chrono::duration<double> prefetch_time = std::chrono::nanoseconds::zero(); // Cumulative time of prefetch sketches to GPU (CUDA UVM)
  std::chrono::duration<double> apply_delta_time = std::chrono::nanoseconds::zero(); // Cumulative time of applying delta sketches (Default)
};