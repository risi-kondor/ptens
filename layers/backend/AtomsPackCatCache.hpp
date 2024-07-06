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

#ifndef _ptens_AtomsPackCatCache
#define _ptens_AtomsPackCatCache

#include "AtomsPackObj.hpp"
#include "AtomsPack.hpp"


namespace ptens{


  //template<typename DUMMY>
  class AtomsPackCatCache: 
    public cnine::plist_indexed_object_bank<AtomsPackObj,shared_ptr<AtomsPackObj>>{
  public:

    typedef cnine::plist_indexed_object_bank<AtomsPackObj,shared_ptr<AtomsPackObj>> BASE;

    AtomsPackCatCache():
      BASE([](const vector<AtomsPackObj*>& v){
	  return shared_ptr<AtomsPackObj>(new AtomsPackObj(AtomsPackObj::cat(v)));}){
    }
      
    AtomsPack operator()(const vector<AtomsPack>& x){
      vector<AtomsPackObj*> v;
      for(auto& p:x)
	v.push_back(p.obj.get());
      return AtomsPack(BASE::operator()(v));
      //if(ptens_global::cache_atomspack_cats) 
      //else return make_shared<AtomsPackObj>(AtomsPackObj::cat(v));
    }

  };

}

#endif 
