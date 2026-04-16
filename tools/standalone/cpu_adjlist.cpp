#include <fstream>
#include <iomanip>
#include <vector>
#include <atomic>
#include <map>
#include <mutex>
#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <edge_store.h>
#include <thread>
#include <limits>

static bool shutdown = false;

class STCPUAdjAlg : public CCSketchAlg {
 private:
  node_id_t *h_edgeUpdates;

  std::map<uint64_t, uint64_t> batch_sizes;
  std::map<uint64_t, uint64_t> batch_src;
  std::map<uint64_t, uint64_t> batch_start_index;
  std::mutex batch_mutex;

  // Atomic variables
  std::atomic<uint64_t> edgeUpdate_offset;
  std::atomic<uint64_t> batch_count;

  // lossless edge storage
  EdgeStore edge_store;

  int num_worker_threads;
  std::chrono::duration<double> ins_time;

 public:
  STCPUAdjAlg(node_id_t num_nodes, size_t num_updates, int num_worker_threads, 
    int num_subgraphs, size_t seed, CCAlgConfiguration config) 
  : CCSketchAlg(num_nodes, seed, config), num_worker_threads(num_worker_threads),
    edge_store(seed, num_nodes, 1, num_subgraphs, 0, false) {

    edgeUpdate_offset = 0;
    batch_count = 0;

    h_edgeUpdates = new node_id_t[2 * num_updates];
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

  void start_adj_insert() {
    size_t num_batches = batch_count;
    std::cout << "Number of batches: " << num_batches << "\n";

    auto ins_start = std::chrono::steady_clock::now();
    auto task = [&](int thr_id) {
      for (int batch_id = thr_id; batch_id < num_batches; batch_id += num_worker_threads) {
        node_id_t src_vertex = batch_src[batch_id];
        size_t update_offset = batch_start_index[batch_id];
        node_id_t* dst_vertices = &h_edgeUpdates[update_offset];

        // We only have an adjacency list so just directly insert
        TaggedUpdateBatch more_upds = edge_store.insert_adj_edges(src_vertex, dst_vertices, batch_sizes[batch_id]);
      }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_worker_threads; i++) threads.emplace_back(task, i);

    // wait for threads to finish
    for (int i = 0; i < num_worker_threads; i++) threads[i].join();
    ins_time = std::chrono::steady_clock::now() - ins_start;
  }

  size_t get_num_edges() { return edge_store.get_num_edges(); }
  std::chrono::duration<double> get_ins_time() { return ins_time; }
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
void track_insertions(uint64_t total, GraphSketchDriver<STCPUAdjAlg> *driver,
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
  int reader_threads = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  BinaryFileStream stream(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  int num_graphs = 1 + (int)(2 * log2(num_nodes));

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  driver_config.gutter_conf().buffer_exp(20).queue_factor(8).wq_batch_per_elm(32);
  auto cc_config = CCAlgConfiguration().batch_factor(1).sketches_factor(k);

  STCPUAdjAlg st_adj_alg{num_nodes, num_updates, num_threads, num_graphs, get_seed(), cc_config};
  GraphSketchDriver<STCPUAdjAlg> driver{&st_adj_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(CONNECTIVITY);

  shutdown = true;
  querier.join();

  st_adj_alg.start_adj_insert();

  std::cout << "Number of edges in adj. list: " << st_adj_alg.get_num_edges() << std::endl;
  std::cout << "Insertion Rate: " << stream.edges() / st_adj_alg.get_ins_time().count() << std::endl;

  std::ofstream out("runtime_results.csv", std::ios_base::out | std::ios_base::app);
  out << std::fixed;
  out << std::setprecision(3);
  out << stream.edges() / st_adj_alg.get_ins_time().count() / 1e6 << std::endl;
}
