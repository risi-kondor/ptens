find_package(Python COMPONENTS Interpreter Development Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ptens_base bindings/ptens_py.cpp)
target_include_directories(ptens_base PUBLIC bindings)
target_link_libraries(ptens_base PUBLIC ptens)

include(${cnine_SOURCE_DIR}/cmake/FindTorchPython.cmake)
find_torch_python_library(TORCH_PYTHON_LIBRARY)
target_link_libraries(ptens_base PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} pybind11::pybind11)

if(WITH_CUDA)
  target_link_libraries(ptens_base PRIVATE CUDA::cudart)
endif(WITH_CUDA)

install(TARGETS ptens_base DESTINATION . COMPONENT ptens::python)
