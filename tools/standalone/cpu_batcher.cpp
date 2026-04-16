#include <fstream>
#include <iomanip>
#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>

static bool shutdown = false;

class EmptySketchAlg : public CCSketchAlg {
 public:
  EmptySketchAlg(node_id_t num_vertices, size_t seed, CCAlgConfiguration config) 
  : CCSketchAlg(num_vertices, seed, config) {};

  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices) {
    if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();

    return;
  };
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
void track_insertions(uint64_t total, GraphSketchDriver<EmptySketchAlg> *driver,
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
    std::cout << "Arguments: stream_file, graph_workers, reader_threads k" << std::endl;
    exit(EXIT_FAILURE);
  }

  shutdown = false;
  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  size_t reader_threads = std::atol(argv[3]);
  int k = std::atoi(argv[4]);

  BinaryFileStream stream(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  driver_config.gutter_conf().buffer_exp(20).queue_factor(8).wq_batch_per_elm(32);
  auto cc_config = CCAlgConfiguration().batch_factor(1).sketches_factor(k);

  EmptySketchAlg empty_alg{num_nodes, get_seed(), cc_config};
  GraphSketchDriver<EmptySketchAlg> driver{&empty_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(CONNECTIVITY);

  shutdown = true;
  querier.join();

  std::chrono::duration<double> insert_time = driver.flush_end - ins_start;
  double num_seconds = insert_time.count();
  std::cout << "Updates per second:           " << stream.edges() / num_seconds << std::endl;

  std::ofstream out("runtime_results.csv", std::ios_base::out | std::ios_base::app);
  out << std::fixed;
  out << std::setprecision(3);
  out << stream.edges() / num_seconds / 1e6 << std::endl;
}
