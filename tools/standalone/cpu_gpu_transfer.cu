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
#include <cuda_kernel.cuh>

static bool shutdown = false;

class STCPUGPUSketchAlg : public CCSketchAlg {
 private:
  SketchParams sketchParams;

  size_t total_bytes;
  Bucket *h_buckets, *d_buckets;
  node_id_t *h_edgeUpdates, *d_edgeUpdates;

  std::map<uint64_t, uint64_t> batch_sizes;
  std::map<uint64_t, uint64_t> batch_src;
  std::map<uint64_t, uint64_t> batch_start_index;
  std::mutex batch_mutex;

  // Atomic variables
  std::atomic<uint64_t> edgeUpdate_offset;
  std::atomic<uint64_t> batch_count;

  int num_worker_threads;

  float trans_time;

 public:
  STCPUGPUSketchAlg(node_id_t num_nodes, size_t num_updates, int num_worker_threads, SketchParams params, CCAlgConfiguration config) 
  : CCSketchAlg(num_nodes, params.seed, config),
    total_bytes(2 * num_updates * sizeof(node_id_t)),
    sketchParams(params),
    num_worker_threads(num_worker_threads) {

    edgeUpdate_offset = 0;
    batch_count = 0;

    gpuErrchk(cudaMallocHost(&h_edgeUpdates, total_bytes));
    gpuErrchk(cudaMalloc(&d_edgeUpdates, total_bytes));
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

  void start_transfer(bool is_hybrid) {
    cudaEvent_t start, stop;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    cudaStream_t s0, s1, s2, s3;

    // Initialize CudaStream
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);

    // Data Transfer
    gpuErrchk(cudaEventRecord(start));
    if (is_hybrid) { // 4N for hybrid
      // Simulate the same amount of memory getting transferred between CPU-GPU
      gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, h_edgeUpdates, total_bytes, cudaMemcpyHostToDevice, s0));
      gpuErrchk(cudaMemcpyAsync(h_edgeUpdates, d_edgeUpdates, total_bytes, cudaMemcpyDeviceToHost, s1));
      gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, h_edgeUpdates, total_bytes, cudaMemcpyHostToDevice, s2));
      gpuErrchk(cudaMemcpyAsync(h_edgeUpdates, d_edgeUpdates, total_bytes, cudaMemcpyDeviceToHost, s3));

      cudaStreamSynchronize(s0);
      cudaStreamSynchronize(s1);
      cudaStreamSynchronize(s2);
      cudaStreamSynchronize(s3);
      gpuErrchk(cudaEventRecord(stop, s3));
    }
    else { // 2N for non-hybrid
      // Simulate the same amount of memory getting transferred between CPU-GPU
      gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, h_edgeUpdates, total_bytes, cudaMemcpyHostToDevice, s0));
      gpuErrchk(cudaMemcpyAsync(h_edgeUpdates, d_edgeUpdates, total_bytes, cudaMemcpyDeviceToHost, s1));

      cudaStreamSynchronize(s0);
      cudaStreamSynchronize(s1);
      gpuErrchk(cudaEventRecord(stop, s1));
    }
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&trans_time, start, stop));
  }

  float get_trans_time() { return trans_time; }
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
void track_insertions(uint64_t total, GraphSketchDriver<STCPUGPUSketchAlg> *driver,
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

  STCPUGPUSketchAlg st_cpugpu_alg{num_nodes, num_updates, num_threads, sketchParams, cc_config};
  GraphSketchDriver<STCPUGPUSketchAlg> driver{&st_cpugpu_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(CONNECTIVITY);

  shutdown = true;
  querier.join();

  // Perform all sketch updates in gpu
  st_cpugpu_alg.start_transfer(is_hybrid);
  double trans_time = st_cpugpu_alg.get_trans_time();

  std::cout << "Transfer Rate: " << stream.edges() / (trans_time * 0.001) << std::endl;

  std::ofstream out("runtime_results.csv", std::ios_base::out | std::ios_base::app);
  out << std::fixed;
  out << std::setprecision(3);
  out << stream.edges() / (trans_time * 0.001) / 1e6 << std::endl;
}
