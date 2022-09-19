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

  //#include "../../bindings/Cgraph_py.cpp"
  //#include "../../bindings/GraphNhoods_py.cpp"

  #include "../../bindings/AtomsPack_py.cpp"
  #include "../../bindings/Hgraph_py.cpp"

  #include "../../bindings/Ptensor0_py.cpp"
  #include "../../bindings/Ptensor1_py.cpp"
  #include "../../bindings/Ptensor2_py.cpp"

  #include "../../bindings/Ptensors0_py.cpp"
  #include "../../bindings/Ptensors1_py.cpp"
  #include "../../bindings/Ptensors2_py.cpp"

  // #include "../../bindings/LinMaps_py.cpp"
  #include "../../bindings/AddMsgFunctions_py.cpp"

  

}
