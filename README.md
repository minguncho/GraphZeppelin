# GPUSketch: CUDA Library for Solving Connectivity and Minimum Cut on Graph Streams
This is the source code of GPUSketch. Our paper on GPUSketch is in submission. 

## Installing and Running GPUSketch
### Requirements
- Unix OS (not Mac, tested on Ubuntu)
- CUDA Toolkit (CUDA 11.8)
- cmake>=3.15

### Installation
1. Clone this repository
2. Create a `build` sub directory at the project root dir.
3. Initialize cmake by running `cmake ..` in the build dir. To include building other executables for tests and tools, run `cmake -DBUILD_TEST_AND_TOOL=ON ..`
4. Build the library and executables for testing by running `cmake --build .` in the build dir.

This library can easily be included with other cmake projects using FetchContent or ExternalProject.

### Main Executables
- `process_stream`: CPU-only graph skething solving the connected components problem.
- `cuda_process_stream`: CPU-GPU graph sketching solving the connected components problem.
- `min_cut`: CPU-GPU graph sketching with hybrid data structure solving the minimum cut problem.

### Running Unit Tests
Run `./tests` from the build directory.
