#include <ascii_file_stream.h>
#include <binary_file_stream.h>
#include <dynamic_erdos_generator.h>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include "cuda_stream.h"
#include "cuda_kernel.cuh"
#include "cc_sketch_alg.h"
#include "graph_sketch_driver.h"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

// Constant global variables
int num_worker_threads = 6;
int num_reader_threads = 6;
int num_device_threads = 1024;
int num_batch_per_buffer = 540;

class TempGPUSketchAlg : public CCSketchAlg {
 private:
  SketchParams sketchParams;
  CudaKernel cudaKernel;

  CudaStream<TempGPUSketchAlg>** cudaStreams;

 public:
  TempGPUSketchAlg(node_id_t num_nodes, SketchParams params, CCAlgConfiguration config) 
  : CCSketchAlg(num_nodes, params.cudaUVM_enabled, params.seed, params.cudaUVM_buckets, config),
    sketchParams(params) {

    cudaStreams = new CudaStream<TempGPUSketchAlg>*[num_worker_threads];
    for (int thr_id = 0; thr_id < num_worker_threads; thr_id++) {
      cudaStreams[thr_id] = new CudaStream<TempGPUSketchAlg>(this, 0, num_nodes, num_device_threads, num_batch_per_buffer, params);
    }

    size_t maxBytes = (params.num_buckets * sizeof(vec_t_cu)) + (params.num_buckets * sizeof(vec_hash_t));
    cudaKernel.updateSharedMemory(maxBytes);

  };

  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices) {
    if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();
    cudaStreams[thr_id]->process_batch(src_vertex, &dst_vertices[0], dst_vertices.size());
  };

  void flush_buffers() {
    for (int thr_id = 0; thr_id < num_worker_threads; thr_id++) {
      cudaStreams[thr_id]->flush_buffers();
    }
    cudaDeviceSynchronize();
  };
};

SketchParams createParams(node_id_t num_nodes, bool cudaUVM_enabled, size_t seed) {
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;
  sketchParams.seed = seed;

  sketchParams.cudaUVM_enabled = cudaUVM_enabled;
  if (cudaUVM_enabled) {
    // Allocate memory for buckets
    Bucket* cudaUVM_buckets;
    gpuErrchk(cudaMallocManaged(&cudaUVM_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket)));
    sketchParams.cudaUVM_buckets = cudaUVM_buckets;
  }

  return sketchParams;
}

// helper function to generate a dynamic binary stream and its cumulative insert only stream
void generate_cc_stream(size_t seed, node_id_t num_vertices, double density, double delete_portion,
                     double adtl_portion, size_t rounds, std::string stream_name) {
  // remove old versions of the stream files
  std::remove(stream_name.c_str());

  // Check if already generated with other unit tests
  if(std::filesystem::exists(stream_name)) return;

  std::cout << "Generating large stream for CC..." << std::endl;

  // generate new stream files
  DynamicErdosGenerator dy_stream(seed, num_vertices, density, delete_portion, adtl_portion,
                                  rounds);
  dy_stream.to_binary_file(stream_name);

}

size_t run_cpu_sketching(BinaryFileStream& stream, size_t seed) {
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  stream.seek(0); // Reset stream

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_worker_threads);
  driver_config.gutter_conf().buffer_exp(20).wq_batch_per_elm(8);
  auto cc_config = CCAlgConfiguration().batch_factor(1);

  CCSketchAlg alg{num_nodes, seed, cc_config};
  GraphSketchDriver<CCSketchAlg> driver{&alg, &stream, driver_config, num_reader_threads};

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(CONNECTIVITY);

  return alg.connected_components().size();
}

size_t run_gpu_sketching(BinaryFileStream& stream, bool UVM_enabled, size_t seed) {
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  stream.seek(0); // Reset stream

  SketchParams params = createParams(num_nodes, UVM_enabled, seed);
  auto config = CCAlgConfiguration().batch_factor(1);
  TempGPUSketchAlg alg(num_nodes, params, config);

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_worker_threads);
  GraphSketchDriver<TempGPUSketchAlg> driver{&alg, &stream, driver_config, num_reader_threads};

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(CONNECTIVITY);
  cudaDeviceSynchronize();
  alg.flush_buffers();
  cudaDeviceSynchronize();
  
  return alg.connected_components().size();
}

TEST(CUDAStreamTestSuite, InitializationTest) {
  node_id_t num_nodes = 8192;
  size_t seed = get_seed();
  SketchParams params = createParams(num_nodes, false, seed);
  auto config = CCAlgConfiguration().batch_factor(1);

  EXPECT_NO_THROW({
    TempGPUSketchAlg alg(num_nodes, params, config);
  });
}

TEST(CUDAStreamTestSuite, ProcessBatchesUVMSmall) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::string stream_file = curr_dir + "/res/multiples_graph_1024_stream.data";

  std::cout << "Processing stream: " << stream_file << std::endl;

  BinaryFileStream stream(stream_file);

  size_t seed = get_seed();
  size_t gpu_ans = run_gpu_sketching(stream, true, seed);

  ASSERT_EQ(78, gpu_ans);
}

TEST(CUDAStreamTestSuite, ProcessBatchesUVMLarge) {
  std::string stream_file = "cc_large_stream";
  generate_cc_stream(get_seed(), 8192, 0.03, 0.5, 0.005, 3, stream_file);
  BinaryFileStream stream(stream_file);

  size_t seed = get_seed();

  size_t cpu_ans = run_cpu_sketching(stream, seed);
  size_t gpu_ans = run_gpu_sketching(stream, true, seed);

  ASSERT_EQ(cpu_ans, gpu_ans);
}

TEST(CUDAStreamTestSuite, ProcessBatchesDefaultSmall) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::string stream_file = curr_dir + "/res/multiples_graph_1024_stream.data";

  std::cout << "Processing stream: " << stream_file << std::endl;

  BinaryFileStream stream(stream_file);

  size_t seed = get_seed();
  size_t gpu_ans = run_gpu_sketching(stream, false, seed);

  ASSERT_EQ(78, gpu_ans);
}

TEST(CUDAStreamTestSuite, ProcessBatchesDefaultLarge) {
  std::string stream_file = "cc_large_stream";
  generate_cc_stream(get_seed(), 8192, 0.03, 0.5, 0.005, 3, stream_file);
  BinaryFileStream stream(stream_file);

  size_t seed = get_seed();

  size_t cpu_ans = run_cpu_sketching(stream, seed);
  size_t gpu_ans = run_gpu_sketching(stream, false, seed);

  ASSERT_EQ(cpu_ans, gpu_ans);
}

