#ifndef _Ptens_base_cpp
#define _Ptens_base_cpp

#include "Cnine_base.cpp"
#include "Ptens_base.hpp"
#include "monitored.hpp"
#include "Ltensor.hpp"


namespace ptens{
  namespace ptens_global{
    cnine::obj_monitor<cnine::Ltensor<int> > atomspack_offsets1_monitor;
    cnine::obj_monitor<cnine::Ltensor<int> > atomspack_offsets2_monitor;
    cnine::obj_monitor<cnine::Ltensor<int> > indexpack_arrg_monitor;
  }
}

#include "AtomsPackCatCache.hpp"
#include "GatherPlanCache.hpp"
#include "CompressedGatherMatrixCache.hpp"

#include "GgraphCache.hpp"
#include "SubgraphCache.hpp"
#include "CompressedSubgraphAtomsPackCache.hpp"


namespace ptens{


  namespace ptens_global{

    bool row_level_operations=false; 
    bool using_pgather=true;

    cnine::MemoryManager* vram_manager=nullptr;

    bool cache_atomspack_cats=true;
    AtomsPackCatCache atomspack_cat_cache; 

    GatherPlanCache gather_plan_cache;
    CompressedGatherMatrixCache gather_matrix_cache;

    GgraphCache graph_cache;
    SubgraphCache subgraph_cache;

    CompressedSubgraphAtomsPackCache c_subgraphatoms_cache;

  }


}


#endif 


  //PtensSessionObj* ptens_session=nullptr;
//#include "PtensSession.hpp"
    //bool cache_overlap_maps=false;
    //OverlapsMmapCache overlaps_cache;

    //bool cache_rmaps=false;
    //RowLevelMapCache rmap_cache;

    //bool cache_subgraph_lists=true;
    //SubgraphListCache subgraph_list_cache;

