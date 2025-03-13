find_package(Python COMPONENTS Interpreter Development Development.Module REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ptens_base bindings/ptens_py.cpp)
target_include_directories(ptens_base PUBLIC bindings)
target_link_libraries(ptens_base PUBLIC ptens)
if(WITH_CUDA)
  target_link_libraries(ptens_base PUBLIC ptens_cuda)
endif(WITH_CUDA)

install(TARGETS ptens_base DESTINATION ptens_base COMPONENT ptens::python)
