#include <fstream>
#include <iomanip>
#include <vector>
#include <atomic>
#include <map>
#include <mutex>
#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>
#include "../../src/cuda_kernel.cu"

static bool shutdown = false;

__global__ void st_sketch_update_kernel(node_id_t* update_src, vec_t* update_sizes, 
    vec_t* update_start_indexes, node_id_t* edgeUpdates, size_t batch_count, size_t num_buckets, 
    size_t num_columns, size_t bkt_per_col, size_t sketchSeed) {

  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_buckets];

  // Each thread will initialize a bucket in shared memory
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }

  __syncthreads();

  // Update sketch - each thread works on 1 update for on 1 column
  int batch_id = blockIdx.x;

  if (batch_id >= batch_count) batch_id -= batch_count; // For hybrid sketching alg, 2x batch count 
  for (int id = threadIdx.x; id < update_sizes[batch_id] * num_columns; id += blockDim.x) {

    int column_id = id % num_columns;
    int update_id = id / num_columns;

    vec_t edge_id = device_concat_pairing_fn(update_src[batch_id], edgeUpdates[update_start_indexes[batch_id] + update_id]);

    vec_hash_t checksum = bucket_get_index_hash(edge_id, sketchSeed);
    
    if ((column_id == 0)) {
      // Update depth 0 bucket
      bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edge_id, checksum);
    }

    // Update higher depth buckets
    col_hash_t depth = bucket_get_index_depth(edge_id, sketchSeed + (column_id * 5), bkt_per_col);
    size_t bucket_id = column_id * bkt_per_col + depth;
    if(depth < bkt_per_col)
      bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edge_id, checksum);
  }
};

class STGPUSketchAlg : public CCSketchAlg {
 private:
  SketchParams sketchParams;
  node_id_t *h_edgeUpdates, *d_edgeUpdates;

  size_t num_updates;

  std::map<uint64_t, uint64_t> batch_sizes;
  std::map<uint64_t, uint64_t> batch_src;
  std::map<uint64_t, uint64_t> batch_start_index;
  std::mutex batch_mutex;

  // Atomic variables
  std::atomic<uint64_t> edgeUpdate_offset;
  std::atomic<uint64_t> batch_count;

  size_t maxBytes;

  float kernel_time;

 public:
  STGPUSketchAlg(node_id_t num_nodes, size_t num_updates, SketchParams params, CCAlgConfiguration config) 
  : CCSketchAlg(num_nodes, params.seed, config),
    sketchParams(params),
    num_updates(num_updates) {

    edgeUpdate_offset = 0;
    batch_count = 0;

    maxBytes = (params.num_buckets * sizeof(vec_t_cu)) +
                      (params.num_buckets * sizeof(vec_hash_t));
    gpuErrchk(cudaFuncSetAttribute(
      st_sketch_update_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes));

    gpuErrchk(cudaMallocHost(&h_edgeUpdates, 2 * num_updates * sizeof(node_id_t)));
    gpuErrchk(cudaMalloc(&d_edgeUpdates, 2 * num_updates * sizeof(node_id_t)));
  };

  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices) {
    if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();

    // Get offset
    size_t offset = edgeUpdate_offset.fetch_add(dst_vertices.size());

    // Fill in buffer
    size_t index = 0;
    for (auto dst : dst_vertices) {
      h_edgeUpdates[offset + index] = dst;
      index++;
    }

    size_t batch_id = batch_count.fetch_add(1);
    std::lock_guard<std::mutex> lk(batch_mutex);
    batch_sizes.insert({batch_id, dst_vertices.size()});
    batch_src.insert({batch_id, src_vertex});
    batch_start_index.insert({batch_id, offset});
  };

  void launch_gpu_kernel(bool is_hybrid) {
    int num_device_threads = 1024;
    int num_device_blocks = is_hybrid ? batch_count.load() * 2 : batch_count.load();

    std::cout << "Num GPU threads per block: " << num_device_threads << "\n";
    std::cout << "Num GPU thread blocks: " << num_device_blocks << "\n";
    std::cout << "Number of batches: " << batch_count << "\n";
    std::cout << "Preparing update buffers for GPU...\n";

    vec_t *h_update_sizes, *d_update_sizes, *h_update_start_index, *d_update_start_index;
    node_id_t *h_update_src, *d_update_src;

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
    gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, 2 * num_updates * sizeof(node_id_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_update_sizes, h_update_sizes, batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_update_src, h_update_src, batch_count * sizeof(node_id_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_update_start_index, h_update_start_index, batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));

    // Launch GPU kernel
    std::cout << "Launching GPU Kernel...\n";

    cudaEvent_t start, stop;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start));
    st_sketch_update_kernel<<<num_device_blocks, num_device_threads, maxBytes>>>(
      d_update_src, d_update_sizes, d_update_start_index, d_edgeUpdates,
      batch_count, sketchParams.num_buckets, sketchParams.num_columns, 
      sketchParams.bkt_per_col, sketchParams.seed);
    gpuErrchk(cudaEventRecord(stop));
    cudaDeviceSynchronize();

    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&kernel_time, start, stop));
    
    std::cout << "  GPU Kernel Finished.\n";
  }

  float get_kernel_time() { return kernel_time; }
};


static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
void track_insertions(uint64_t total, GraphSketchDriver<STGPUSketchAlg> *driver,
                      std::chrono::steady_clock::time_point start_time) {
  total = total * 2; // we insert 2 edge updates per edge

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev  = start_time;
  uint64_t prev_updates = 0;

  while(true) {
    sleep(1);
    uint64_t updates = driver->get_total_updates();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff   = now - prev;

    // calculate the insertion rate
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    size_t ins_per_sec = (((double)(upd_delta)) / cur_diff.count()) / 2;

    if (updates >= total || shutdown)
      break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(), std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %lu per second\r", progress * 5, ins_per_sec); fflush(stdout);
  }

  printf("Progress:====================| Done                             \n");
  return;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads k is_hybrid" << std::endl;
    exit(EXIT_FAILURE);
  }

  shutdown = false;
  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  int reader_threads = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  bool is_hybrid;
  if (std::string(argv[5]) == "true") {
    is_hybrid = true;
    std::cout << "Hybrid sketch alg. enabled\n";
  }
  else if (std::string(argv[5]) == "false") {
    is_hybrid = false;
  }   
  else {
    std::cout << "Invalid option for is_hybrid: " << argv << "\n";
    exit(EXIT_FAILURE); 
  }

  BinaryFileStream stream(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  // Get variables from sketch
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, k);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  std::cout << "num_samples: " << sketchParams.num_samples << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n"; 

  // Getting sketch seed
  sketchParams.seed = get_seed();

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  driver_config.gutter_conf().buffer_exp(20).queue_factor(8).wq_batch_per_elm(32);
  auto cc_config = CCAlgConfiguration().batch_factor(1).sketches_factor(k);

  STGPUSketchAlg st_gpu_alg{num_nodes, num_updates, sketchParams, cc_config};
  GraphSketchDriver<STGPUSketchAlg> driver{&st_gpu_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(CONNECTIVITY);

  shutdown = true;
  querier.join();

  // Perform all sketch updates in gpu
  st_gpu_alg.launch_gpu_kernel(is_hybrid);
  double update_time = st_gpu_alg.get_kernel_time();

  std::cout << "  Sketch Update rate: " << num_updates / (update_time * 0.001) << std::endl;

  std::ofstream out("runtime_results.csv", std::ios_base::out | std::ios_base::app);
  out << std::fixed;
  out << std::setprecision(3);
  out << stream.edges() / (update_time * 0.001) / 1e6 << std::endl;
}
