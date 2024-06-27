#ifndef _Ptens_base_cpp
#define _Ptens_base_cpp

#include "Cnine_base.cpp"
#include "Ptens_base.hpp"

namespace ptens{

  //class AtomsPackCatCache;

  namespace ptens_global{
    //bool cache_atomspack_cats=true;
    //AtomsPackCatCache* atomspack_cat_cache=nullptr;
  }

}

//#include "AtomsPack.hpp"
#include "AtomsPackCatCache.hpp"
#include "OverlapsMmapCache.hpp"
#include "RowLevelMapCache.hpp"

//#include "PtensSessionObj.hpp"

namespace ptens{


  namespace ptens_global{

    bool row_level_operations=false; 

    bool cache_atomspack_cats=true;
    AtomsPackCatCache atomspack_cat_cache; 
    //atomspack_cat_cache=new AtomsPackCatCache(); 

    bool cache_overlap_maps=false;
    OverlapsMmapCache overlaps_cache;

    bool cache_rmaps=false;
    RowLevelMapCache rmap_cache;

  }


}


#endif 


  //PtensSessionObj* ptens_session=nullptr;
//#include "PtensSession.hpp"
