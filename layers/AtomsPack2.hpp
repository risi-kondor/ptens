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

#ifndef _ptens_AtomsPack0
#define _ptens_AtomsPack0

#include "AtomsPack.hpp"
#include "AtomsPack2obj.hpp"
#include "CompoundTransferMap.hpp"

namespace ptens{

  class AtomsPack2{
  public:


    shared_ptr<AtomsPack2obj<int> > obj;


  public: // ---- Maps ---------------------------------------------------------------------------------------
    
    
    template<typename SOURCE>
    GatherMapProgram overlaps_transfer_map(const SOURCE& x){
      return CompoundTransferMap(obj->overlaps_transfer_map(*x.obj));
    }


  };

}

#endif 
