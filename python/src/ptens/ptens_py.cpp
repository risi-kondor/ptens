#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "Cgraph.hpp"
#include "GraphNhoods.hpp"
#include "Ptensor0.hpp"
#include "Ptensor0pack.hpp"
#include "Ptensor1pack.hpp"
#include "Ptensor2pack.hpp"


//std::default_random_engine rndGen;

//GElib::GElibSession session;


#include "Cnine_base.cpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace ptens;
  namespace py=pybind11;

  #include "../../bindings/Cgraph_py.cpp"
  #include "../../bindings/GraphNhoods_py.cpp"

  #include "../../bindings/Ptensor0_py.cpp"
  #include "../../bindings/Ptensor1_py.cpp"
  #include "../../bindings/Ptensor2_py.cpp"

  #include "../../bindings/Ptensor0pack_py.cpp"
  #include "../../bindings/Ptensor1pack_py.cpp"
  #include "../../bindings/Ptensor2pack_py.cpp"


  

}
