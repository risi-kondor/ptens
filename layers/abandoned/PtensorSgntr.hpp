/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */
#ifndef _ptens_PtensorSgntr
#define _ptens_PtensorSgntr

#include "Ptens_base.hpp"


namespace ptens{

  class PtensorSgntr{
  public:

    int k;
    int nc;
    
    PtensorSgntr(const int _k, const int _nc): k(_k), nc(_nc){}

    bool operator==(const PtensorSgntr& x) const{
      return (k==x.k)&&(nc==x.nc);
    }
    
  };

}

namespace std{
  template<>
  struct hash<ptens::PtensorSgntr>{
  public:
    size_t operator()(const ptens::PtensorSgntr& sgntr) const{
      return (hash<int>()(sgntr.k)<<1)^hash<int>()(sgntr.nc); 
    }
  };
}



#endif 
