#include <chrono>
#include <cmath>
#include <vector>

#include <sketch.h>
#include "../src/cuda_kernel.cu"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

__global__ void gpuSketchTest_kernel(int num_device_blocks, node_id_t num_nodes, size_t num_updates, size_t num_buckets, Bucket* buckets, size_t num_columns, size_t bkt_per_col, size_t sketchSeed) {

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
}


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes num_updates" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "SKETCH COMPUTE THROUGHPUT TEST - GPU:\n";

  int device_id = cudaGetDevice(&device_id);
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  std::cout << "-----CUDA Device Information-----\n";
  std::cout << "CUDA Device Count: " << device_count << "\n";
  std::cout << "CUDA Device ID: " << device_id << "\n";
  std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n"; 
  std::cout << "CUDA Max. Shared memory per Block: " << (double)deviceProp.sharedMemPerBlockOptin / 1000 << "KB\n";

  size_t free_memory;
  size_t total_memory;

  cudaMemGetInfo(&free_memory, &total_memory);
  std::cout << "GPU Free (Available) Memory: " << (double)free_memory / 1000000000 << "GB\n";
  std::cout << "GPU Total Memory: " << (double)total_memory / 1000000000 << "GB\n";
  std::cout << "\n";

  node_id_t num_nodes = std::atoi(argv[1]);
  size_t num_updates = std::stoull(argv[2]);

  // Single Sketch that fills up entire GPU memory
  /*SketchParams sketchParams;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_columns = (free_memory / sizeof(Bucket)) / sketchParams.bkt_per_col;
  sketchParams.num_columns -= 100000; // Reduce some columns so not using 100% GPU memory (still using nearly 100%)
  //sketchParams.num_columns /= 4;
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;*/
  
  // Single Sketch with size corresponding to num_nodes
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  // Single Sketch with size corresponding to GPU SM
  /*SketchParams sketchParams;
  sketchParams.num_columns = deviceProp.multiProcessorCount * 64;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;*/

  std::cout << "-----Sketch Information-----\n";
  std::cout << "num_nodes: " << num_nodes << "\n";
  std::cout << "num_updates: " << num_updates << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "\n";

  int num_device_threads = 1024;
  int num_updates_per_blocks = (sketchParams.num_buckets * sizeof(Bucket)) / sizeof(node_id_t);
  int num_device_blocks = std::ceil((double)num_updates / num_updates_per_blocks);

  std::cout << "Batch Size: " << num_updates_per_blocks << "\n\n";

  /*int *num_tb_columns;
  gpuErrchk(cudaMallocManaged(&num_tb_columns, num_device_blocks * sizeof(int)));

  for (int i = 0; i < num_device_blocks ; i++) {
    num_tb_columns[i] = sketchParams.num_columns / num_device_blocks;
  }

  // If num_columns doesn't get divided evenly
  size_t leftover_num_columns = sketchParams.num_columns - ((sketchParams.num_columns / num_device_blocks) * num_device_blocks);
  int k_id = num_device_blocks - 1;
  while (leftover_num_columns > 0) {
    num_tb_columns[k_id]++;
    k_id--;
    leftover_num_columns--;
  }*/

  //size_t num_columns_per_block = (deviceProp.sharedMemPerBlockOptin / sizeof(Bucket)) / sketchParams.bkt_per_col;
  /*size_t num_columns_per_block = 64;
  /size_t num_buckets_per_block = num_columns_per_block * sketchParams.bkt_per_col + 1; 
  size_t maxBytes = num_buckets_per_block * sizeof(vec_t_cu) + num_buckets_per_block * sizeof(vec_hash_t);*/

  //size_t num_last_tb_buckets = (num_tb_columns[num_device_blocks - 1] * sketchParams.bkt_per_col) + 1;
  //size_t maxBytes = (num_last_tb_buckets * sizeof(vec_t_cu)) + (num_last_tb_buckets * sizeof(vec_hash_t));
  size_t maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t));
  cudaFuncSetAttribute(gpuSketchTest_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);

  std::cout << "-----GPU Kernel Information-----\n";
  std::cout << "Number of thread blocks: " << num_device_blocks << "\n";
  std::cout << "Number of threads per block: " << num_device_threads << "\n";
  std::cout << "Memory Size for buckets: " << (double)(num_nodes * sketchParams.num_buckets * sizeof(Bucket)) / 1000000000 << "GB\n";
  //std::cout << "Number of columns per thread block: " << num_columns_per_block << "\n";
  /*std::cout << "Number of columns of each thread block: ";
  for (int i = 0; i < num_device_blocks ; i++) {
    std::cout << num_tb_columns[i] << ", ";
  }
  std::cout << "\n";*/
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

  cudaFree(d_buckets);
}
