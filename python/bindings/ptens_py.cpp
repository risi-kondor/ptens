#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "Cnine_base.cpp"

#include "PtensSession.hpp"

#include "Hgraph.hpp"
#include "PtensFindPlantedSubgraphs.hpp"
#include "Ggraph.hpp"
#include "Subgraph.hpp"

#include "BatchedGgraph.hpp"
#include "BatchedAtomsPack.hpp"

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Tensor.hpp"
#include "ATview.hpp"

#include "LinmapFunctions.hpp"
#include "MsgFunctions.hpp"

#include "GatherLayers.hpp"
#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"
#include "OuterLayers.hpp"
// //#include "ConcatLayers.hpp"

#include "NodeLayer.hpp"
#include "SubgraphLayer0.hpp"
#include "SubgraphLayer1.hpp"
#include "SubgraphLayer2.hpp"

#include "Ptensors0b.hpp"
#include "Ptensors1b.hpp"
#include "Ptensors2b.hpp"

//#include "NodeLayerb.hpp"
#include "SubgraphLayer0b.hpp"
#include "SubgraphLayer1b.hpp"
#include "SubgraphLayer2b.hpp"

#include "BatchedPtensors0b.hpp"


namespace ptens{ 
  PtensSession ptens_session;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace ptens;
  namespace py=pybind11;
  

#include "AtomsPack_py.cpp"
#include "Hgraph_py.cpp"
#include "Ggraph_py.cpp"
#include "Subgraph_py.cpp"
  
#include "Ptensor0_py.cpp"
#include "Ptensor1_py.cpp"
#include "Ptensor2_py.cpp"
  
#include "Ptensors0_py.cpp"
#include "Ptensors1_py.cpp"
#include "Ptensors2_py.cpp"

#include "LinmapFunctions_py.cpp"
#include "MsgFunctions_py.cpp"
#include "OuterFunctions_py.cpp"

#include "NodeLayer_py.cpp"
#include "SubgraphLayer0_py.cpp"
#include "SubgraphLayer1_py.cpp"
#include "SubgraphLayer2_py.cpp"

#include "Ptensors0b_py.cpp"
#include "Ptensors1b_py.cpp"
#include "Ptensors2b_py.cpp"

    //#include "NodeLayerb_py.cpp"
#include "SubgraphLayer0b_py.cpp"
#include "SubgraphLayer1b_py.cpp"
#include "SubgraphLayer2b_py.cpp"

#include "BatchedGgraph_py.cpp"
#include "BatchedAtomsPack_py.cpp"

#include "BatchedPtensors0b_py.cpp"


}
