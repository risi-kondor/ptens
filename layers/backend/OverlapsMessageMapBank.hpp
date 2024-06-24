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

#ifndef _ptens_OverlapsMessageMapBank
#define _ptens_OverlapsMessageMapBank

#include "AtomsPackObj.hpp"
#include "TransferMapObj.hpp"


namespace ptens{

  typedef AtomsPackObj DUMMYC;
  //extern PtensSessionObj* ptens_session;
  extern bool cache_overlap_maps;


  class OverlapsMessageMapBank: 
    public cnine::ptr_pair_indexed_object_bank<AtomsPackObj,AtomsPackObj,shared_ptr<TransferMapObj<DUMMYC> > >{
  public:

    typedef cnine::ptr_pair_indexed_object_bank<AtomsPackObj,AtomsPackObj,shared_ptr<TransferMapObj<DUMMYC> > > BASE;

    OverlapsMessageMapBank():
      BASE([](const AtomsPackObj& out, const AtomsPackObj& in){
	  return shared_ptr<TransferMapObj<DUMMYC> >(new TransferMapObj<DUMMYC> (in,out));}){}


  public: // ---- Access ------------------------------------------------------------------------------------------


    shared_ptr<TransferMapObj<DUMMYC> > operator()(const AtomsPackObj& out, const AtomsPackObj& in){
      if(cache_overlap_maps) return BASE::operator()(out,in); 
      return shared_ptr<TransferMapObj<DUMMYC> >(new TransferMapObj<DUMMYC>(in,out));
    }


  };

}

#endif 
