#include <mutex>
#include <set>
#include <vector>

#include "cuda_kernel.cuh"
#include "cuda_stream.h"

enum GraphType {
  SKETCH = 0,
  ADJLIST = 1,
  FIXED_ADJLIST = 2
};

class MCSubgraph {
 private:
  int graph_id;
  CudaUpdateParams* cudaUpdateParams;
  std::atomic<GraphType> type;
  node_id_t num_nodes;

  int num_host_threads;  // Number of CPU threads

  std::atomic<int> conversion_counter;

  std::atomic<size_t> num_sketch_updates;
  std::atomic<size_t> num_adj_edges;

  std::vector<std::set<node_id_t>> adjlist;

  CudaStream** cudaStreams;

 public:
  std::mutex* adj_mutex;

  // Constructor
  MCSubgraph(int graph_id, node_id_t num_nodes, int num_host_threads, int num_device_threads, int num_batch_per_buffer,
             CudaUpdateParams* cudaUpdateParams, GraphType type, size_t batch_size, size_t sketchSeed);
  MCSubgraph(int graph_id, node_id_t num_nodes, int num_host_threads, GraphType type);
  ~MCSubgraph();

  void insert_adj_edge(node_id_t src, std::vector<node_id_t> dst_vertices);
  void insert_sketch_buffer(int thr_id, node_id_t src, std::vector<node_id_t> dst_vertices);
  void flush_sketch_buffers();

  // Get methods
  CudaUpdateParams* get_cudaUpdateParams() { return cudaUpdateParams; }
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