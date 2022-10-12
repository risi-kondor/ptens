#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "Cnine_base.cpp"

//#include "Cgraph.hpp"
//#include "GraphNhoods.hpp"
#include "PtensSession.hpp"
#include "Hgraph.hpp"
#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "LinmapFunctions.hpp"
#include "MsgFunctions.hpp"
#include "OuterFunctions.hpp"


ptens::PtensSession session;

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

  #include "LinmapFunctions_py.cpp"
  #include "MsgFunctions_py.cpp"
  #include "OuterFunctions_py.cpp"

}
