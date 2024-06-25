#include "Cnine_base.cpp"
#include "Ptens_base.hpp"
#include "OverlapsMmapCache.hpp"

//#include "PtensSessionObj.hpp"

#ifndef _Ptens_base_cpp
#define _Ptens_base_cpp

namespace ptens{


  namespace ptens_global{

    bool row_level_operations=false; 

    bool cache_overlap_maps=false;
    OverlapsMmapCache overlaps_cache;

  }


}


#endif 


  //PtensSessionObj* ptens_session=nullptr;
//#include "PtensSession.hpp"
