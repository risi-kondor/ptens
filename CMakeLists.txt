cmake_minimum_required(VERSION 3.25)
project( ptens VERSION 0.5.0 LANGUAGES CXX CUDA)
include(CheckLanguage)

option(WITH_CUDA ON)
option(COPY_WARNING OFF)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WITH_CUDA)
  check_language(CUDA)
  # Set the C++ standard
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED CUDA::cudart CUDA::cublas)

  else(CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA compiler found")
    set(WITH_CUDA OFF)
  endif(CMAKE_CUDA_COMPILER)
endif(WITH_CUDA)

# Enable CPM
include(cmake/CPM.cmake)

find_package(Torch REQUIRED)
cpmaddpackage(NAME cnine GITHUB_REPOSITORY InnocentBug/cnine GIT_TAG cmake)
find_package(cnine REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_library(ptens cuda/dummy.cpp)

set(ptens_include_dirs include caches tensors layers layers/backend batched batched/backend compressed compressed/backend)

target_include_directories(ptens PUBLIC ${ptens_include_dirs})
target_link_libraries(ptens PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(ptens PUBLIC cnine)


target_compile_definitions(ptens PUBLIC _WITH_ATEN)
target_compile_definitions(ptens PUBLIC CNINE_RANGE_CHECKING)
target_compile_definitions(ptens PUBLIC CNINE_SIZE_CHECKING)
target_compile_definitions(ptens PUBLIC CNINE_DEVICE_CHECKING)
target_compile_definitions(ptens PUBLIC WITH_FAKE_GRAD)
target_compile_definitions(ptens PUBLIC CNINE_FUNCTION_TRACING)

if(WITH_CUDA)
  target_compile_definitions(ptens PUBLIC _WITH_CUDA)
  target_compile_definitions(ptens PUBLIC _WITH_CUBLAS)
  target_compile_definitions(ptens PUBLIC WITH_FAKE_GRAD)
  target_compile_definitions(ptens PUBLIC -D_DEF_CGCMEM)
  target_compile_options(ptens PUBLIC --default-stream=per-thread)
endif(WITH_CUDA)

target_compile_options(ptens PUBLIC -Wno-sign-compare)
target_compile_options(ptens PUBLIC -Wno-deprecated-declarations)
target_compile_options(ptens PUBLIC -Wno-unused-variable)
target_compile_options(ptens PUBLIC -Wno-unused-but-set-variable)
target_compile_options(ptens PUBLIC -Wno-reorder)
target_compile_options(ptens PUBLIC -Wno-reorder-ctor)
target_compile_options(ptens PUBLIC -Wno-overloaded-virtual)

if(WITH_CUDA)
  add_subdirectory(cuda)
endif(WITH_CUDA)

option(BUILD_PYTHON "Build the python bindings" ON)
if(BUILD_PYTHON)
  add_subdirectory(python)
endif(BUILD_PYTHON)
