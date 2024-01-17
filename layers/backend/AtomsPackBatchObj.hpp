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

#ifndef _Ptens_AtomsPackBatchObj
#define _Ptens_AtomsPackBatchObj

#include "object_pack_s.hpp"
#include "AtomsPackObj.hpp"
#include "MessageListBatch.hpp"


namespace ptens{


  class AtomsPackBatchObj: object_pack_s<AtomsPackObj>{
  public:

    typedef object_pack_s<AtomsPackObj> BASE;

    using BASE::size;
    using BASE::operator[];


    cnine::ptr_indexed_object_bank<AtomsPackBatchObj,MessageListBatch> overlaps_mlist=
      cnine::ptr_indexed_object_bank<AtomsPackBatchObj,MessageListBatch>([this](const AtomsPackBatchObj& _y){
	  MessageListBatch R;
	  zip(_y,[&](const AtomsPackObj& x, const AtomsPackObj& y){
	      R.obj.push_back(x.overlaps_mlist(y).obj)});
	  return R;
	}

  };

}

#endif 
