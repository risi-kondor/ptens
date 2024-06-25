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


#ifndef _ptens_AtomsPackTag
#define _ptens_AtomsPackTag

#include "AtomsPackObj.hpp"


namespace ptens{

  class AtomsPackTag: public cnine::observable<AtomsPackTag>{
  public:

    weak_ptr<AtomsPackObj> _atoms;

    //operator shared_ptr<AtomsPackObj>(){
    //return _atoms.lock();
    //}

    shared_ptr<AtomsPackObj> atoms() const{
      return _atoms.lock();
    }

    AtomsPackObj& operator*() const{
      return *_atoms.lock();
    }

    AtomsPackObj* operator->() const{
      return _atoms.lock().get();
    }

  protected:

    AtomsPackTag(const shared_ptr<AtomsPackObj>& x):
      observable(this),
      _atoms(x){}
      
  };


  class AtomsPackTag0: public AtomsPackTag{
  public:

    using AtomsPackTag::AtomsPackTag;

    static shared_ptr<AtomsPackTag0> make(const shared_ptr<AtomsPackObj>& x){
      if(!x->cached_tag0) x->cached_tag0=shared_ptr<AtomsPackTag0>(new AtomsPackTag0(x));
      return x->cached_tag0;
    }

  };
}

#endif 
