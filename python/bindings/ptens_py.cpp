#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

//#include "Cgraph.hpp"
//#include "GraphNhoods.hpp"
#include "Hgraph.hpp"
#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "LinMaps.hpp"
#include "AddMsgFunctions.hpp"


//std::default_random_engine rndGen;

//GElib::GElibSession session;


#include "Cnine_base.cpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace ptens;
  namespace py=pybind11;
  

  #include "AtomsPack_py.cpp"
  #include "Hgraph_py.cpp"

  #include "Ptensor0_py.cpp"
  #include "Ptensor1_py.cpp"
  #include "Ptensor2_py.cpp"

  #include "Ptensors0_py.cpp"
  #include "Ptensors1_py.cpp"
  #include "Ptensors2_py.cpp"

  #include "Linmaps_py.cpp"
  #include "AddMsgFunctions_py.cpp"

}