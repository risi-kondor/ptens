#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "Cgraph.hpp"
#include "GraphNhoods.hpp"

//std::default_random_engine rndGen;

//GElib::GElibSession session;


#include "Cnine_base.cpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace ptens;
  namespace py=pybind11;

  #include "../../bindings/Cgraph_py.cpp"


  

}
