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


namespace ptens{

  typedef AtomsPackObj DUMMYC;


  namespace ptens_global{
    extern bool cache_rmaps;
  }


  class RowLevelMmapCache: 
    public cnine::ptr_pair_indexed_object_bank<PtensorJigs,PtensorsJig,TensorLevelMapObj,shared_ptr<RowLevelMap> >{
  public:

    typedef cnine::ptr_pair_indexed_object_bank<PtensorJig,PtensorJig,TensorLevelMapObj,shared_ptr<RowLevelMap> > BASE;

    OverlapsMmapCache():
      BASE([](const PtensorJig& out, const PtensorJig& in){
	  return shared_ptr<TensorLevelMapObj>(new TensorLevelMapObj(in,out));}){}


  public: // ---- Access ------------------------------------------------------------------------------------------


    shared_ptr<TensorLevelMapObj> operator()(const PtensorJig& out, const PtensorJig& in){
      if(ptens_global::cache_row_level_mmaps) return BASE::operator()(out,in); 
      return shared_ptr<TensorLevelMapObj>(new TensorLevelMapObj(in,out));
    }



    shared_ptr<TensorLevelMapObj<DUMMYC> > dispatch(const PtensorJig& out, const PtensorJig& in, const TensorLevelMap& map){
      


  };

}

#endif 
