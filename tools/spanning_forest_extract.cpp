#include <string.h>
#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <cmath>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

std::pair<double, double> calc_stats(const std::vector<size_t> &data) {
  double avg = 0;
  size_t total = 0;
  for (size_t i = 0; i < data.size(); i++) {
    avg += i * data[i];
    total += data[i];
  }
  avg /= total;

  double dev = 0;
  for (size_t i = 0; i < data.size(); i++) {
    dev += data[i] * pow(double(i) - avg, 2);
  }

  return {avg, sqrt(dev / total)};
}

class TempCCSketchAlg : public CCSketchAlg {
 private:

 public:
  TempCCSketchAlg(node_id_t num_nodes, size_t seed, CCAlgConfiguration config) 
  : CCSketchAlg(num_nodes, seed, config) {};

  std::vector<SpanningForest> calc_disjoint_spanning_forests(size_t k) {
    std::vector<SpanningForest> SFs;

    for (size_t i = 0; i < k; i++) {
      SFs.push_back(calc_spanning_forest());
      filter_sf_edges(SFs[SFs.size() - 1]);
    }

    // revert the state of the sketches to remove all deletions
    for (auto &sf : SFs) {
      filter_sf_edges(sf);
    }

    return SFs;
  }
};

// Main function that populates a graph and then tests how many SFs can be extracted from it
int main(int argc, char **argv) {
  if (argc < 3 || argc > 5) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_stream trials [seed]" << std::endl;
    std::cout << "\"graph_stream\" must be a BinaryFileStream." << std::endl;
    std::cout << "Optionally specify a seed, otherwise one is chosen randomly" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_file = argv[1];
  size_t trials = std::stol(argv[2]);
  size_t seed = get_seed();
  if (argc == 5) {
    seed = std::stol(argv[3]);
  }

  size_t num_threads = 40;
  size_t reader_threads = 8;
  std::vector<size_t> rounds_required;
  std::mt19937_64 seed_gen(seed);

  node_id_t num_vertices;
  size_t num_updates;
  {
    BinaryFileStream stream(stream_file);
    num_vertices = stream.vertices();
    num_updates  = stream.edges();
  }
  size_t vertex_power = ceil(log2(num_vertices));
  size_t errors = 0;
  size_t empty = 0;
  std::cout << "Extracting " << vertex_power << " Spanning Forests from: " << stream_file << std::endl;
  std::cout << "vertices    = " << num_vertices << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  for (size_t trial = 0; trial < trials; trial++) {
    BinaryFileStream stream(stream_file);
    std::cout << "Trial #" << trial << std::endl;

    auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
    auto cc_config = CCAlgConfiguration().sketches_factor(1.33);
    TempCCSketchAlg cc_alg{num_vertices, seed_gen(), cc_config};
    GraphSketchDriver<TempCCSketchAlg> driver{&cc_alg, &stream, driver_config, reader_threads};

    rounds_required.resize(cc_alg.max_rounds());

    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query(KSPANNINGFORESTS);

    // figure out how many rounds are required to extract log V spanning forests
    cc_alg.calc_disjoint_spanning_forests(vertex_power);

    // add number of rounds to get log V spanning forests to vector
    rounds_required[cc_alg.last_query_rounds] += 1;
  }

  std::cout << std::endl;
  std::cout << "ERRORS = " << errors << std::endl;
  std::cout << "EMPTY = " << empty << std::endl;

  auto stats = calc_stats(rounds_required);
  std::cout << "avg = " << stats.first << " std dev = " << stats.second << std::endl;

  std::ofstream out("runtime_results.csv", std::ios_base::out | std::ios_base::app);
  for (size_t i = 0; i < rounds_required.size(); i++) {
    out << i << ", " << rounds_required[i] << std::endl;
  }
}