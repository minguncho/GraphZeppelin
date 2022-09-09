#include <graph.h>
 #include <binary_graph_stream.h>
 
 std::string file_name = "/mnt/ssd2/binary_streams/kron_17_stream_binary";
 
 int main() {
  BinaryGraphStream stream(file_name, 1024*32);
  node_id_t num_nodes   = stream.nodes();
  size_t    num_updates = stream.edges();
  auto config = GraphConfiguration().gutter_sys(CACHETREE).num_groups(46);
  Graph g{num_nodes, config};

  auto start = std::chrono::steady_clock::now();
  for (size_t e = 0; e < num_updates; e++)       // Loop through all the updates in the stream
    g.update(stream.get_edge());                 // Update the graph by applying the next edge update

  auto CC_num= g.connected_components().size();
  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
  double num_seconds = diff.count();
  printf("Total insertion time was: %lf\n", num_seconds);
  printf("Insertion rate was:       %lf\n", stream.edges() / num_seconds);
}