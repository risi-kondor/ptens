#include "Cnine_base.cpp"
#include "PtensSessionObj.hpp"

#ifndef _Ptens_base_cpp
#define _Ptens_base_cpp

namespace ptens{

  class OverlapsMessageMapBank;


  PtensSessionObj* ptens_session=nullptr;

  bool cache_overlap_maps=false;
  OverlapsMessageMapBank* overlaps_bank;

}

#include "PtensSession.hpp"

#endif 
