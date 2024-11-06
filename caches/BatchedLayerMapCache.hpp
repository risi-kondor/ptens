/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 */

#ifndef _ptens_BatchedLayerMapCache
#define _ptens_BatchedLayerMapCache

#include "ptr_pair_indexed_object_bank.hpp"
#include "BatchedAtomsPackObj.hpp"
#include "BatchedLayerMapObj.hpp"


namespace ptens{

  class BatchedLayerMapCache: 
    public cnine::ptr_pair_indexed_object_bank<BatchedAtomsPackObj,BatchedAtomsPackObj,shared_ptr<BatchedLayerMapObj> >{
  public:


  };

}

#endif 

