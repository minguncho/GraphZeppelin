#include <fstream>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <graph_sketch_driver.h>
#include <mc_standalone_gpu_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>
#include <sys/resource.h> // for rusage
#include <cuda_kernel.cuh>

static bool shutdown = false;
static bool cudaUVM_enabled = false;

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}

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
void track_insertions(uint64_t total, GraphSketchDriver<MCStandaloneGPUSketchAlg> *driver,
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
  if (argc != 5) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads, k" << std::endl;
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
  int num_batch_per_buffer = 540;

  BinaryFileStream stream(stream_file);
  node_id_t num_vertices = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_vertices << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  std::cout << "k: " << k << std::endl;
  std::cout << "Num batches per buffer: " << num_batch_per_buffer << std::endl;

  int num_graphs = 1 + (int)(2 * log2(num_vertices));
  std::cout << "Total num_graphs: " << num_graphs << "\n";

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  auto mc_config = CCAlgConfiguration().batch_factor(1);

  // Get variables from sketch
  // (1) num_samples (2) num_columns (3) bkt_per_col (4) num_buckets
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_vertices, k);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_vertices));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;
  sketchParams.sharedmem_enabled = true;

  std::cout << "num_samples: " << sketchParams.num_samples << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n"; 

  // Total bytes of sketching datastructure of one subgraph
  int w = 4; // 4 bytes when num_vertices < 2^32
  double sketch_bytes = 4 * w * num_vertices * ((2 * log2(num_vertices)) + 2) * ((k * log2(num_vertices))/(1 - log2(1.2)));
  double adjlist_edge_bytes = 8;

  std::cout << "Total bytes of sketching data structure of one subgraph: " << sketch_bytes / 1000000000 << "GB\n";

  // Calculate number of minimum adj. list subgraph
  size_t num_edges_complete = (size_t(num_vertices) * (size_t(num_vertices) - 1)) / 2;
  int max_sketch_graphs = 0;

  for (int i = 0; i < num_graphs; i++) {
    // Calculate estimated memory for current subgraph
    size_t num_est_edges = num_edges_complete / (1 << i);
    double adjlist_bytes = adjlist_edge_bytes * num_est_edges;

    if (adjlist_bytes >= sketch_bytes) {
      max_sketch_graphs++;
    }
  }

  // Total number of estimated edges of minimum number of adj. list graphs
  double total_sketch_bytes = sketch_bytes * max_sketch_graphs;

  std::cout << "  If complete graph with current num_vertices..." << "\n";
  std::cout << "    Maximum number of sketch graphs: " << max_sketch_graphs << "\n";
  std::cout << "    Total minimum memory required for maximum number of sketch graphs: " << total_sketch_bytes / 1000000000 << "GB\n";

  size_t initial_sketch_graphs = 1;
  if (max_sketch_graphs < initial_sketch_graphs) {
    std::cerr << "WARNING: max_sketch_graphs < initial_sketch_graphs. " << max_sketch_graphs << " < " << initial_sketch_graphs << std::endl; 
    std::cerr << "  Setting initial_sketch_graphs as 0" << std::endl;
    initial_sketch_graphs = 0;
  }

  // FORCING THIS, MAKE SURE TO REMOVE
  initial_sketch_graphs = 0;

  // Reconfigure sketches_factor based on reduced_k
  mc_config.sketches_factor(k);

  std::cout << "CUDA UVM Enabled: " << cudaUVM_enabled << "\n";
  sketchParams.cudaUVM_enabled = cudaUVM_enabled;

  // Getting sketch seed
  sketchParams.seed = get_seed();
  std::cout << "Sketch Seed: " << sketchParams.seed << "\n";
  MCStandaloneGPUSketchAlg mc_standalone_gpu_alg{num_vertices, num_updates, num_threads, reader_threads, num_batch_per_buffer, sketchParams, 
    num_graphs, max_sketch_graphs, initial_sketch_graphs, k, sketch_bytes, mc_config};

  GraphSketchDriver<MCStandaloneGPUSketchAlg> driver{&mc_standalone_gpu_alg, &stream, driver_config, reader_threads};

  std::cout << "Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;

  auto collect_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, collect_start);

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(KSPANNINGFORESTS);

  auto collect_end = std::chrono::steady_clock::now();

  shutdown = true;
  querier.join();

  // Inserting edges in batches to hybrid data structure
  auto ins_start = std::chrono::steady_clock::now();

  mc_standalone_gpu_alg.insert_start();
  mc_standalone_gpu_alg.apply_flush_updates();

  auto ins_end = std::chrono::steady_clock::now();

  mc_standalone_gpu_alg.print_subgraph_edges();

  // Perform query to check answer
  /*double og_k = ((double)k * log2(num_vertices)) / 1.5;
  double eps = std::sqrt((1.5 * log2(num_vertices)) / og_k);

  std::cout << "eps: " << eps << "\n";
  std::cout << "og_k: " << og_k << "\n";*/

  std::chrono::duration<double> collect_time = collect_end - collect_start;
  std::chrono::duration<double> insert_time = ins_end - ins_start;

  std::cout << "Edge collection time(sec): " << collect_time.count() << std::endl;
  std::cout << "  Updates per second: " << stream.edges() / collect_time.count() << std::endl;
  std::cout << "Insertion time(sec): " << insert_time.count() << std::endl;
  std::cout << "  Updates per second: " << stream.edges() / insert_time.count() << std::endl;
  std::cout << "Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;
}
