cmake_minimum_required(VERSION 3.15)
project(BipartiteDatasets LANGUAGES CXX)

include (FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

FetchContent_Declare(
  Eigen

  GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
  GIT_TAG         3.4.0
  SOURCE_SUBDIR   "MADE-UP-DIRECTORY"
)

FetchContent_MakeAvailable(Eigen)

add_executable(example_exe
  example_exe.cpp
)
target_include_directories(example_exe SYSTEM PRIVATE "${FETCHCONTENT_BASE_DIR}/eigen-src")
