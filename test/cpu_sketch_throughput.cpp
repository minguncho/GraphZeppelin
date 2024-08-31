#pragma once
#include <chrono>
#include <thread>
#include <vector>

#include <sketch.h>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

/*__global__ void gpuSketchTest_kernel(int num_device_blocks, node_id_t num_nodes, size_t num_updates, size_t num_buckets, Bucket* buckets, size_t num_columns, size_t bkt_per_col, size_t sketchSeed) {

  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_buckets];

  for (size_t i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }

  __syncthreads();

  size_t update_offset = num_updates * num_columns * blockIdx.x;
  node_id_t node_id = blockIdx.x / num_nodes;
  for (size_t id = threadIdx.x; id < num_updates * num_columns; id += blockDim.x) {

    size_t column_id = (update_offset + id) % num_columns;
    size_t update_id = (update_offset + id) / num_columns;

    // Get random edge id based on current update_id
    //vec_t edge_id = update_id % num_nodes;
    vec_t edge_id = device_concat_pairing_fn(node_id, update_id % num_nodes);

    vec_hash_t checksum = bucket_get_index_hash(edge_id, sketchSeed);
    
    if ((blockIdx.x == num_device_blocks - 1)  && (column_id == 0)) {
      // Update depth 0 bucket
      bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edge_id, checksum);
    }

    // Update higher depth buckets
    col_hash_t depth = bucket_get_index_depth(edge_id, sketchSeed + ((column_id) * 5), bkt_per_col);
    size_t bucket_id = column_id * bkt_per_col + depth;
    if(depth < bkt_per_col)
      bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edge_id, checksum);
  }

  __syncthreads();

  for (size_t i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    atomicXor((vec_t_cu*)&buckets[(node_id * num_buckets) + i].alpha, bucket_a[i]);
    atomicXor((vec_t_cu*)&buckets[(node_id * num_buckets) + i].gamma, (vec_t_cu)bucket_c[i]);
  }
}*/


int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes num_updates num_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  node_id_t num_nodes = std::atoi(argv[1]);
  size_t num_updates = std::stoull(argv[2]);
  int num_threads = std::atoi(argv[3]);

  size_t sketchSeed = get_seed();

  size_t num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  size_t num_columns = num_samples * Sketch::default_cols_per_sample;
  size_t bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  size_t num_buckets = num_columns * bkt_per_col + 1;

  std::cout << "-----Sketch Information-----\n";
  std::cout << "num_nodes: " << num_nodes << "\n";
  std::cout << "num_updates: " << num_updates << "\n";
  std::cout << "bkt_per_col: " << bkt_per_col << "\n";
  std::cout << "num_columns: " << num_columns << "\n";
  std::cout << "num_buckets: " << num_buckets << "\n";
  std::cout << "\n";

  int num_updates_per_batch = (num_buckets * sizeof(Bucket)) / sizeof(node_id_t);
  int num_batches = std::ceil((double)num_updates / num_updates_per_batch);

  std::cout << "Number of Batches: " << num_batches << "\n";
  std::cout << "Batch Size: " << num_updates_per_batch << "\n";

  Sketch **delta_sketches = new Sketch *[num_threads];
  for (size_t thr_id = 0; thr_id < num_threads; thr_id++) {
    delta_sketches[thr_id] = new Sketch(Sketch::calc_vector_length(num_nodes), sketchSeed, Sketch::calc_cc_samples(num_nodes, 1));
  }

  auto sketch_update_start = std::chrono::steady_clock::now();

  auto task = [&](int thr_id) {
    for (int batch_id = thr_id; batch_id < num_batches; batch_id += num_threads) {
      // Reset delta sketch
      delta_sketches[thr_id]->zero_contents();

      node_id_t src_vertex = batch_id / num_nodes;

      for (int update_id = 0; update_id < num_updates_per_batch; update_id++) {
        delta_sketches[thr_id]->update(static_cast<vec_t>(concat_pairing_fn(src_vertex, update_id % num_nodes)));
      }
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; i++) threads.emplace_back(task, i);

  // wait for threads to finish
  for (size_t i = 0; i < num_threads; i++) threads[i].join();

  std::chrono::duration<double> sketch_update_duration = std::chrono::steady_clock::now() - sketch_update_start;
  std::cout << "Total insertion time(sec):    " << sketch_update_duration.count() << std::endl;
  std::cout << "Updates per second:           " << num_updates / sketch_update_duration.count() << std::endl;

  /*
  int num_device_threads = 1024;
  int num_updates_per_blocks = num_device_threads;
  int num_device_blocks = num_updates / num_device_threads;

  size_t maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t));
  cudaFuncSetAttribute(gpuSketchTest_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);

  std::cout << "-----GPU Kernel Information-----\n";
  std::cout << "Number of thread blocks: " << num_device_blocks << "\n";
  std::cout << "Number of threads per block: " << num_device_threads << "\n";
  std::cout << "Memory Size for buckets: " << (double)(num_nodes * sketchParams.num_buckets * sizeof(Bucket)) / 1000000000 << "GB\n";
  std::cout << "  Allocated Shared Memory of: " << (double)maxBytes / 1000 << "KB\n";
  std::cout << "\n";

  Bucket* d_buckets;
  gpuErrchk(cudaMalloc(&d_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket)));

  size_t sketchSeed = get_seed();

  auto sketch_update_start = std::chrono::steady_clock::now();
  gpuSketchTest_kernel<<<num_device_blocks, num_device_threads, maxBytes>>>(num_device_blocks, num_nodes, num_updates_per_blocks, sketchParams.num_buckets, d_buckets, sketchParams.num_columns, sketchParams.bkt_per_col, sketchSeed);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
  std::chrono::duration<double> sketch_update_duration = std::chrono::steady_clock::now() - sketch_update_start;

  std::cout << "Total insertion time(sec):    " << sketch_update_duration.count() << std::endl;
  std::cout << "Updates per second:           " << num_updates / sketch_update_duration.count() << std::endl;

  cudaFree(d_buckets);*/
}
