cmake_minimum_required(VERSION 3.15)
project(GPUSketch LANGUAGES CXX CUDA)

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

include (FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(CMAKE_CUDA_ARCHITECTURES "70")

# Make the default build type Release. If user or another
# project sets a different value than use that
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to default -- Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
endif()
message(STATUS "GPUSketch Build Type: ${CMAKE_BUILD_TYPE}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message(STATUS "Adding GNU compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
elseif(STATUS "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message("Adding MSVC compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  message(STATUS "${CMAKE_CXX_COMPILER_ID} not recognized, no flags added")
endif()

# Check if this project is the top directory or build type is Debug
# If so, build executables, otherwise, only build libraries
get_directory_property(not_root PARENT_DIRECTORY)
if (not_root AND "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(BUILD_EXE OFF)
else()
  set(BUILD_EXE ON)
  message (STATUS "GPUSketch building executables")
endif()

# Print out information of CUDA compiler
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")


# Get GutterTree Project
FetchContent_Declare(
  GutterTree

  GIT_REPOSITORY  https://github.com/GraphStreamingProject/GutterTree.git
  GIT_TAG         main
)

# Get StreamingUtilities
FetchContent_Declare(
  StreamingUtilities

  GIT_REPOSITORY  https://github.com/GraphStreamingProject/StreamingUtilities.git
  GIT_TAG         main
)

# Get VieCut
FetchContent_Declare(
  VieCut

  GIT_REPOSITORY  https://github.com/etwest/VieCut.git
  GIT_TAG         master
)

# Get tlx
FetchContent_Declare(
  tlx

  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         master
)

if (BUILD_BENCH)
  # Get Google Benchmark
  FetchContent_Declare(
    benchmark

    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG        v1.6.1
  )
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

  FetchContent_MakeAvailable(benchmark)
endif()

FetchContent_MakeAvailable(GutterTree StreamingUtilities VieCut tlx)

# AVAILABLE COMPILATION DEFINITIONS:
# VERIFY_SAMPLES_F   Use a deterministic connected-components
#                    algorithm to verify post-processing.
# NO_EAGER_DSU       Do not use the eager DSU query optimization
#                    if this flag is present.
# L0_SAMPLING        Run the CubeSketch l0 sampling algorithm
#                    to ensure that we sample uniformly.
#                    Otherwise, run a support finding algorithm.
#
# Example:
# cmake -DCMAKE_CXX_FLAGS="-DL0_SAMPLING" ..

# Turning off eager DSU for GPUSKetch by default
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_EAGER_DSU")

add_library(CUDALibrary
  src/cuda_library.cu
  src/cuda_kernel.cu
  src/cc_gpu_sketch_alg.cpp
  src/mc_gpu_sketch_alg.cpp
  src/mc_standalone_gpu_sketch_alg.cpp
  src/sk_gpu_sketch_alg.cpp)
add_dependencies(CUDALibrary GutterTree StreamingUtilities VieCut tlx)
target_link_libraries(CUDALibrary PUBLIC xxhash GutterTree StreamingUtilities VieCut tlx)
target_include_directories(CUDALibrary PUBLIC include/)
target_compile_options(CUDALibrary PUBLIC -lineinfo) # -lineinfo: For profiling

add_library(GPUSketch
  src/cc_sketch_alg.cpp
  src/mc_sketch_alg.cpp
  src/edge_store.cpp
  src/return_types.cpp
  src/driver_configuration.cpp
  src/cc_alg_configuration.cpp
  src/sketch.cpp
  src/util.cpp)
add_dependencies(GPUSketch GutterTree StreamingUtilities VieCut tlx)
target_link_libraries(GPUSketch PUBLIC xxhash GutterTree StreamingUtilities VieCut tlx)
target_include_directories(GPUSketch PUBLIC include/)
target_compile_options(GPUSketch PUBLIC -fopenmp)
target_link_options(GPUSketch PUBLIC -fopenmp)
target_compile_definitions(GPUSketch PUBLIC XXH_INLINE_ALL)

add_library(GPUSketchVerifyCC
  src/cc_sketch_alg.cpp
  src/edge_store.cpp
  src/return_types.cpp
  src/driver_configuration.cpp
  src/cc_alg_configuration.cpp
  src/sketch.cpp
  src/util.cpp
  test/util/graph_verifier.cpp)
add_dependencies(GPUSketchVerifyCC GutterTree StreamingUtilities VieCut)
target_link_libraries(GPUSketchVerifyCC PUBLIC xxhash GutterTree StreamingUtilities VieCut)
target_include_directories(GPUSketchVerifyCC PUBLIC include/ include/test/)
target_compile_options(GPUSketchVerifyCC PUBLIC -fopenmp)
target_link_options(GPUSketchVerifyCC PUBLIC -fopenmp)
target_compile_definitions(GPUSketchVerifyCC PUBLIC XXH_INLINE_ALL VERIFY_SAMPLES_F)

if (BUILD_EXE)
  add_executable(tests
    test/test_runner.cpp
    test/cc_alg_test.cpp
    test/sketch_test.cpp
    test/edge_store_test.cpp
    test/dsu_test.cpp
    test/util_test.cpp
    test/util/graph_verifier_test.cpp
    test/xxhash_test.cu)
  add_dependencies(tests GPUSketchVerifyCC)
  target_link_libraries(tests PRIVATE GPUSketchVerifyCC)

  add_executable(statistical_sketch_test
    tools/sketch_testing.cpp)
  add_dependencies(statistical_sketch_test GPUSketchVerifyCC)
  target_link_libraries(statistical_sketch_test PRIVATE GPUSketchVerifyCC)
 
  # executable for processing a binary graph stream
  add_executable(process_stream
    tools/process_stream.cpp)
  target_link_libraries(process_stream PRIVATE GPUSketch)

  # executable for performing in depth correctness testing
  add_executable(test_correctness
    tools/test_correctness.cpp)
  target_link_libraries(test_correctness PRIVATE GPUSketchVerifyCC)

  # executable for processing a binary graph stream for CUDA
  add_executable(cuda_process_stream
    tools/cuda_process_stream.cu)
  target_link_libraries(cuda_process_stream PRIVATE GPUSketch CUDALibrary)

  add_executable(min_cut
    tools/min_cut.cu)
  target_link_libraries(min_cut PRIVATE GPUSketch CUDALibrary)
endif()

if (BUILD_TEST_AND_TOOL)
  add_executable(mc_standalone
    tools/mc_standalone.cu)
  target_link_libraries(mc_standalone PRIVATE GPUSketch CUDALibrary)

  add_executable(single_kernel_stream
    tools/single_kernel_stream.cu)
  target_link_libraries(single_kernel_stream PRIVATE GPUSketch CUDALibrary)

  add_executable(gpu_sketch_throughput
    tools/throughput/gpu_sketch_throughput.cu)
  target_link_libraries(gpu_sketch_throughput PRIVATE GPUSketch CUDALibrary)

  add_executable(gpu_sketch_throughput_density
    tools/throughput/gpu_sketch_throughput_density.cu)
  target_link_libraries(gpu_sketch_throughput_density PRIVATE GPUSketch CUDALibrary)
  
  add_executable(gpu_sketch_throughput_stream
    tools/throughput/gpu_sketch_throughput_stream.cu)
  target_link_libraries(gpu_sketch_throughput_stream PRIVATE GPUSketch CUDALibrary)

  add_executable(cpu_sketch_throughput
    tools/throughput/cpu_sketch_throughput.cpp)
  target_link_libraries(cpu_sketch_throughput PRIVATE GPUSketch)

  add_executable(cpu_sketch_throughput_density
    tools/throughput/cpu_sketch_throughput_density.cpp)
  target_link_libraries(cpu_sketch_throughput_density PRIVATE GPUSketch)

  add_executable(cpu_sketch_throughput_stream
    tools/throughput/cpu_sketch_throughput_stream.cpp)
  target_link_libraries(cpu_sketch_throughput_stream PRIVATE GPUSketch CUDALibrary)

  add_executable(cpu_gpu_transfer
    tools/cpu_gpu_transfer.cu)
  target_link_libraries(cpu_gpu_transfer PRIVATE GPUSketch CUDALibrary)

  add_executable(adjlist_test
    tools/adjlist_test.cpp)
  target_link_libraries(adjlist_test PRIVATE GPUSketch CUDALibrary)

  add_executable(generate_stream
    tools/stream_generators/non_trivial_stream_generator.cpp)
  target_link_libraries(generate_stream PRIVATE GPUSketch)

  add_executable(csv_converter
    tools/converter/csv_converter.cpp)
  target_link_libraries(csv_converter PRIVATE GPUSketch)

  add_executable(metis_converter
    tools/converter/metis_converter.cpp)
  target_link_libraries(metis_converter PRIVATE GPUSketch)

  add_executable(edgelist_converter
    tools/converter/edgelist_converter.cpp)
  target_link_libraries(edgelist_converter PRIVATE GPUSketch)

  add_executable(stream_converter
    tools/converter/stream_converter.cpp)
  target_link_libraries(stream_converter PRIVATE GPUSketch)

  add_executable(convert_bipartite_graph
    tools/converter/convert_bipartite_graph.cpp)
  target_link_libraries(convert_bipartite_graph PRIVATE GPUSketch)

  add_executable(stream_shuffler
    tools/others/stream_shuffler.cpp)
  target_link_libraries(stream_shuffler PRIVATE GPUSketch)

  add_executable(mincut_estimator
    tools/others/mincut_estimator.cpp)
  target_link_libraries(mincut_estimator PRIVATE GPUSketch)

  add_executable(mincut_count
    tools/others/mincut_count.cpp)
  target_link_libraries(mincut_count PRIVATE GPUSketch)
endif()

if (BUILD_BENCH)
  add_executable(bench_cc
    tools/benchmark/graphcc_bench.cpp)
  add_dependencies(bench_cc GPUSketch benchmark)
  target_link_libraries(bench_cc GPUSketch benchmark::benchmark xxhash)
endif()

