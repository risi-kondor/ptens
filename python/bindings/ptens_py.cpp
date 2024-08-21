#include <torch/torch.h>
#include <pybind11/stl.h>

#include "Ptens_base.cpp"
#include "SimpleMemoryManager.hpp"

#include "PtensSession.hpp"
#include "Ggraph.hpp"
#include "Subgraph.hpp"

#include "LayerMap.hpp"
#include "GgraphPreloader.hpp"

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"

#include "SubgraphLayer1.hpp"

#include "BatchedGgraph.hpp"
#include "BatchedAtomsPack.hpp"
#include "BatchedLayerMap.hpp"

#include "BatchedPtensors0.hpp"
#include "BatchedPtensors1.hpp"
#include "BatchedPtensors2.hpp"

#include "CompressedPtensors1.hpp"
#include "CompressedPtensors2.hpp"


ptens::PtensSession ptens_session(1); // Ltensors are not thread safe 


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  using namespace cnine;
  using namespace ptens;
  namespace py=pybind11;

  //typedef cnine::Ltensor<float> tensorf;
  typedef cnine::TensorView<float> tensorf;

  typedef Ptensor0<float> Ptensor0f;
  typedef Ptensor1<float> Ptensor1f;
  typedef Ptensor2<float> Ptensor2f;

  typedef Ptensors0<float> Ptensors0f;
  typedef Ptensors1<float> Ptensors1f;
  typedef Ptensors2<float> Ptensors2f;

  typedef SubgraphLayer1<float> SGlayer1f;

  typedef BatchedLayerMap BLmap;

  typedef BatchedPtensors0<float> BPtensors0f;
  typedef BatchedPtensors1<float> BPtensors1f;
  typedef BatchedPtensors2<float> BPtensors2f;

  typedef CompressedPtensors1<float> CPtensors1f;
  typedef CompressedPtensors2<float> CPtensors2f;


#include "PtensGlobal_py.cpp"

#include "AtomsPack_py.cpp"
#include "AindexPack_py.cpp"
#include "CatomsPack_py.cpp"

#include "Ggraph_py.cpp"
#include "Subgraph_py.cpp"
#include "LayerMap_py.cpp"
#include "GgraphPreloader_py.cpp"

#include "SimpleMemoryManager_py.cpp"
#include "GgraphCache_py.cpp"
#include "SubgraphCache_py.cpp"
#include "CSubgraphatomsCache_py.cpp"

#include "Ptensor0_py.cpp"
#include "Ptensor1_py.cpp"
#include "Ptensor2_py.cpp"
  
#include "Ptensors0_py.cpp"
#include "Ptensors1_py.cpp"
#include "Ptensors2_py.cpp"

#include "SubgraphLayer1_py.cpp"

#include "BatchedGgraph_py.cpp"
#include "BatchedAtomsPack_py.cpp"
#include "BatchedLayerMap_py.cpp"

#include "BatchedPtensors0_py.cpp"
#include "BatchedPtensors1_py.cpp"
#include "BatchedPtensors2_py.cpp"

#include "CPtensors1_py.cpp"
#include "CPtensors2_py.cpp"

    }
