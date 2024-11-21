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

#ifndef _ptens_BatchedAindexPack
#define _ptens_BatchedAindexPack

#include "object_pack_s.hpp"
#include "AindexPack.hpp"

namespace ptens{


  class BatchedAindexPack: public cnine::object_pack_s<AindexPack>{
  public:

    typedef cnine::object_pack_s<AindexPack> BASE;

    using BASE::BASE;

    int count1=0;
    int count2=0;

    
  };
  
}


#endif 
