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

#ifndef _ptens_PtensorsShared
#define _ptens_PtensorsShared

#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"
#include "ptr_pair_indexed_object_bank.hpp"
#include "observable.hpp"


namespace ptens{

  template<typename TYPE>
  class Ptensors0b;

  template<typename OWNER>
  class PtensorsShared: public cnine::observable<PtensorsShared<OWNER> >{
  public:

    typedef cnine::ptr_pair_indexed_object_bank<MessageListObj,PtensorsShared,MessageMap> MMBank;


    MMBank row_map0=MMBank([&](const MessageListObj& lists, const AtomsPackObjBase& atoms){
	return OWNER::row_map0(lists,atoms);});
    
    MMBank row_map1=MMBank([&](const MessageListObj& lists, const AtomsPackObjBase& atoms){
	return OWNER::row_map1(lists,atoms);});
    
    MMBank row_map2=MMBank([&](const MessageListObj& lists, const AtomsPackObjBase& atoms){
	return OWNER::row_map2(lists,atoms);});


    template<typename TYPE>
    MessageMap rmap0(const Ptensors0b<TYPE>& y, const MessageList& lists){
      return row_map0(*lists.obj,*y.shared);
    }

  };

}

#endif 
