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

include(cmake/CPM.cmake)

find_package(Torch REQUIRED)

cpmfindpackage(NAME cnine GITHUB_REPOSITORY InnocentBug/cnine GIT_TAG cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_library(ptens STATIC cuda/dummy.cpp)
set_target_properties(ptens PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(ptens_include_dirs include caches tensors layers layers/backend batched batched/backend compressed compressed/backend)

foreach(directory IN LISTS ptens_include_dirs)
  target_include_directories(ptens PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${directory}> $<INSTALL_INTERFACE:${directory}>)
endforeach()
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

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ptensConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ptensConfig.cmake  INSTALL_DESTINATION lib/cmake/ptens)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ptensVersion.cmake  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

# Install the library
install(TARGETS ptens  EXPORT ptens  LIBRARY DESTINATION lib  ARCHIVE DESTINATION lib INCLUDES DESTINATION ptens/include)

# Install the headers
foreach(directory IN LISTS ptens_include_dirs)
  install(DIRECTORY ${directory}/ DESTINATION ${directory})
endforeach()

# Install the config files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ptensConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/ptensVersion.cmake  DESTINATION lib/cmake/ptens)

# Install the export
install(EXPORT ptens FILE ptensTargets.cmake NAMESPACE ptens:: DESTINATION lib/cmake/ptens)

# Register package in user's package registry
export(PACKAGE ptens)
