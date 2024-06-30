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
 */

#ifndef _ptens_OverlapsMmapCache
#define _ptens_OverlapsMmapCache

#include "AtomsPack.hpp"
#include "AtomsPackObj.hpp"
#include "TensorLevelMapObj.hpp"
#include "TensorLevelMap.hpp"


namespace ptens{

  namespace ptens_global{
    extern bool cache_overlap_maps;
  }

  class OverlapsMmapCache: 
    public cnine::ptr_pair_indexed_object_bank<AtomsPackObj,AtomsPackObj,shared_ptr<TensorLevelMapObj> >{
  public:

    typedef cnine::ptr_pair_indexed_object_bank<AtomsPackObj,AtomsPackObj,shared_ptr<TensorLevelMapObj> > BASE;

    OverlapsMmapCache():
      BASE([](const AtomsPackObj& out, const AtomsPackObj& in){
	  return shared_ptr<TensorLevelMapObj>(new TensorLevelMapObj (in,out));}){}


  public: // ---- Access ------------------------------------------------------------------------------------------


    TensorLevelMap operator()(const AtomsPackObj& out, const AtomsPackObj& in){
      if(ptens_global::cache_overlap_maps) return BASE::operator()(out,in); 
      return new TensorLevelMapObj(in,out);
    }

    TensorLevelMap operator()(const AtomsPack& out, const AtomsPack& in){
      return (*this)(*out.obj,*in.obj);
    }

  };

}

#endif 
