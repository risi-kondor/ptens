#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

//#include "Cnine_base.cpp"
#include "Ptens_base.cpp"

//#include "SimpleMemoryManager.hpp"
//#include "Ltensor.hpp"
#include "ATview.hpp"

//#include "Hgraph.hpp"
#include "Ggraph.hpp"
#include "Subgraph.hpp"

//#include "BatchedGgraph.hpp"
//#include "BatchedAtomsPack.hpp"

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"

//#include "LinmapFunctions.hpp"
//#include "MsgFunctions.hpp"

//#include "GatherLayers.hpp"
//#include "LinmapLayers.hpp"
//#include "EMPlayers.hpp"
//#include "OuterLayers.hpp"
// //#include "ConcatLayers.hpp"

//#include "NodeLayer.hpp"
//#include "SubgraphLayer0.hpp"
//#include "SubgraphLayer1.hpp"
//#include "SubgraphLayer2.hpp"

/*
#include "SubgraphLayer0b.hpp"
#include "SubgraphLayer1b.hpp"
#include "SubgraphLayer2b.hpp"

#include "BatchedPtensors0b.hpp"
#include "BatchedPtensors1b.hpp"
#include "BatchedPtensors2b.hpp"

#include "BatchedSubgraphLayer0b.hpp"
#include "BatchedSubgraphLayer1b.hpp"
#include "BatchedSubgraphLayer2b.hpp"
*/

ptens::PtensSession ptens_session(8);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  using namespace cnine;
  using namespace ptens;
  namespace py=pybind11;

  typedef cnine::Ltensor<float> tensorf;
  
  typedef Ptensor0<float> Ptensor0f;
  typedef Ptensor1<float> Ptensor1f;
  typedef Ptensor2<float> Ptensor2f;

  typedef Ptensors0<float> Ptensors0f;
  typedef Ptensors1<float> Ptensors1f;
  typedef Ptensors2<float> Ptensors2f;

#include "PtensGlobal_py.cpp"

#include "AtomsPack_py.cpp"
//#include "Hgraph_py.cpp"
#include "Ggraph_py.cpp"
#include "Subgraph_py.cpp"

#include "AindexPack_py.cpp"

    //#include "MessageList_py.cpp"
    //#include "AtomsPack0_py.cpp"
    //#include "MessageMap_py.cpp"
  
#include "Ptensor0_py.cpp"
#include "Ptensor1_py.cpp"
#include "Ptensor2_py.cpp"
  
#include "Ptensors0_py.cpp"
#include "Ptensors1_py.cpp"
#include "Ptensors2_py.cpp"

    //#include "LinmapFunctions_py.cpp"
    //#include "MsgFunctions_py.cpp"
    //#include "OuterFunctions_py.cpp"

    //#include "NodeLayer_py.cpp"
    //#include "SubgraphLayer0_py.cpp"
    //#include "SubgraphLayer1_py.cpp"
    //#include "SubgraphLayer2_py.cpp"

    //#include "Ptensors0_py.cpp"
    //#include "Ptensors1_py.cpp"
    //#include "Ptensors2_py.cpp"

/*
#include "SubgraphLayer0b_py.cpp"
#include "SubgraphLayer1b_py.cpp"
#include "SubgraphLayer2b_py.cpp"

#include "BatchedGgraph_py.cpp"
#include "BatchedAtomsPack_py.cpp"

#include "BatchedPtensors0b_py.cpp"
#include "BatchedPtensors1b_py.cpp"
#include "BatchedPtensors2b_py.cpp"

#include "BatchedSubgraphLayer0b_py.cpp"
#include "BatchedSubgraphLayer1b_py.cpp"
#include "BatchedSubgraphLayer2b_py.cpp"
    */

}
