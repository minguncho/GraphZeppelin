#include <mutex>
#include <set>
#include <vector>

#include "mc_gpu_sketch_alg.h"
#include "cuda_kernel.cuh"
#include "cuda_stream.h"

enum GraphType {
  SKETCH = 0,
  ADJLIST = 1,
  FIXED_ADJLIST = 2
};

template <class Alg>
class MCSubgraph {
 private:
  Alg *sketching_alg;
  int graph_id;
  SketchParams sketchParams;
  std::atomic<GraphType> type;
  node_id_t num_nodes;

  int num_host_threads;  // Number of CPU threads

  std::atomic<int> conversion_counter;

  std::atomic<size_t> num_sketch_updates;
  std::atomic<size_t> num_adj_edges;

  std::vector<std::set<node_id_t>> adjlist;

  CudaStream<Alg>** cudaStreams;

 public:
  std::mutex* adj_mutex;

  // Constructor
  MCSubgraph(Alg *sketching_alg, int graph_id, node_id_t num_nodes, int num_host_threads, int num_device_threads, int num_batch_per_buffer,
             SketchParams _sketchParams, GraphType type)
    : sketching_alg(sketching_alg),
      graph_id(graph_id),
      num_nodes(num_nodes),
      num_host_threads(num_host_threads),
      type(type),
      adjlist(num_nodes) {
    conversion_counter = 0;

    num_sketch_updates = 0;
    num_adj_edges = 0;

    adj_mutex = new std::mutex[num_nodes];

    // Rewrite address for buckets 
    sketchParams = _sketchParams;
    if (sketchParams.cudaUVM_enabled) {
      sketchParams.cudaUVM_buckets = &sketchParams.cudaUVM_buckets[graph_id * num_nodes * sketchParams.num_buckets];
    }
    
    // Initialize CUDA Streams
    cudaStreams = new CudaStream<Alg>*[num_host_threads];
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      cudaStreams[thr_id] = new CudaStream<Alg>(sketching_alg, graph_id, num_nodes, num_device_threads, num_batch_per_buffer, sketchParams);
    }
  }

  MCSubgraph(int graph_id, node_id_t num_nodes, int num_host_threads, GraphType type)
    : graph_id(graph_id),
    num_host_threads(num_host_threads),
    type(type),
    num_nodes(num_nodes),
    adjlist(num_nodes) {

    num_adj_edges = 0;

    adj_mutex = new std::mutex[num_nodes];
  }

  ~MCSubgraph() {
    delete[] adj_mutex;
  }

  void insert_adj_edge(node_id_t src, std::vector<node_id_t> dst_vertices) {
    std::lock_guard<std::mutex> lk(adj_mutex[src]);
    int num_updated_edges = 0;
    for (auto dst : dst_vertices) {
      if (adjlist[src].find(dst) == adjlist[src].end()) {
        adjlist[src].insert(dst);
        num_updated_edges++;
      } else {
        adjlist[src].erase(dst);  // Current edge already exist, so delete
        num_updated_edges--;
      }
    }
    num_adj_edges += num_updated_edges;
  }

  void insert_sketch_buffer(int thr_id, node_id_t src, std::vector<node_id_t> dst_vertices) {
    num_sketch_updates += dst_vertices.size();
    cudaStreams[thr_id]->process_batch(src, &dst_vertices[0], dst_vertices.size());
  }

  void flush_sketch_buffers() {
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      cudaStreams[thr_id]->flush_buffers();
    }
  }

  // Get methods
  SketchParams& get_sketchParams() { return sketchParams; }
  GraphType get_type() { return type; }
  size_t get_num_updates() {
    if (type == SKETCH) {
      return num_sketch_updates;
    } else {
      return num_adj_edges;
    }
  }
  size_t get_num_adj_edges() { return num_adj_edges; }
  const std::vector<std::set<node_id_t>>& get_adjlist() { return adjlist; }
  const std::set<node_id_t>& get_neighbor_nodes(node_id_t src) { return adjlist[src]; }

  bool try_acq_conversion() {
    int org_val = 0;
    int new_val = 1;
    return conversion_counter.compare_exchange_strong(org_val, new_val);
  }

  // Set methods
  void set_type(GraphType new_type) { type = new_type; }
  void increment_num_sketch_updates(int value) { num_sketch_updates += value; }
};