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

# Fetch nlohmann/json
FetchContent_Declare(
  json

  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)

FetchContent_MakeAvailable(Eigen json)

find_package(OpenMP REQUIRED)

add_executable(example_exe.o
  example_exe.cpp
)
target_include_directories(example_exe.o SYSTEM PRIVATE "${FETCHCONTENT_BASE_DIR}/eigen-src")

add_executable(data_parser
  data_parser.cpp
)
target_link_libraries(data_parser PRIVATE nlohmann_json::nlohmann_json)

add_executable(convokit_parser
  convokit_parser.cpp
)
target_link_libraries(convokit_parser PRIVATE nlohmann_json::nlohmann_json)

add_executable(convokit_overlap_finder
  convokit_overlap_finder.cpp
)
target_link_libraries(convokit_overlap_finder PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

add_executable(convokit_upperbound
  convokit_upperbound.cpp
)
target_link_libraries(convokit_upperbound PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

add_executable(convokit_num_users
  convokit_num_users.cpp
)
target_link_libraries(convokit_num_users PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
