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


  class AtomsPackTagObj: public cnine::observable<AtomsPackTagObj>{
  public:

    friend class AtomsPackTag0; 
    friend class AtomsPackTag1; 
    friend class AtomsPackTag2; 

    weak_ptr<AtomsPackObj> atoms;

    const AtomsPackObj& get_atoms() const{
      return *atoms.lock();
    }

    //AtomsPackObj& operator*() const{
    //return *_atoms.lock();
    //}

    //AtomsPackObj* operator->() const{
    //return _atoms.lock().get();
    //}

  protected:

    AtomsPackTagObj(const shared_ptr<AtomsPackObj>& x):
      observable(this),
      atoms(x){}
      
  };


  class AtomsPackTagObj0: public AtomsPackTagObj{
  public:
    using AtomsPackTagObj::AtomsPackTagObj;
  };


  class AtomsPackTagObj1: public AtomsPackTagObj{
  public:
    using AtomsPackTagObj::AtomsPackTagObj;
  };

  class AtomsPackTagObj2: public AtomsPackTagObj{
  public:
    using AtomsPackTagObj::AtomsPackTagObj;
  };


  // ---- Tags ------------------------------------------------------------------------------------------------------
  

  class AtomsPackTag0{
  public:

    shared_ptr<AtomsPackTagObj0> obj;

    AtomsPackTag0(const AtomsPack& x):
      AtomsPackTag0(x.obj){}

    AtomsPackTag0(const shared_ptr<AtomsPackObj>& x){
      if(!x->cached_tag0) x->cached_tag0=shared_ptr<AtomsPackTagObj0>(new AtomsPackTagObj0(x));
      obj=x->cached_tag0;
    }
    
    //AtomsPackObj& operator*() const{
    //return *(obj->atoms.lock());
    //}

    //AtomsPackObj* operator->() const{
    //return obj->atoms.lock().get();
    //}
    
  };


  class AtomsPackTag1{
  public:

    shared_ptr<AtomsPackTagObj1> obj;

    AtomsPackTag1(const AtomsPack& x):
      AtomsPackTag1(x.obj){}

    AtomsPackTag1(const shared_ptr<AtomsPackObj>& x){
      if(!x->cached_tag1) x->cached_tag1=shared_ptr<AtomsPackTagObj1>(new AtomsPackTagObj1(x));
      obj=x->cached_tag1;
    }

  };


  class AtomsPackTag2{
  public:

    shared_ptr<AtomsPackTagObj2> obj;

    AtomsPackTag2(const AtomsPack& x):
      AtomsPackTag2(x.obj){}

    AtomsPackTag2(const shared_ptr<AtomsPackObj>& x){
      if(!x->cached_tag2) x->cached_tag2=shared_ptr<AtomsPackTagObj2>(new AtomsPackTagObj2(x));
      obj=x->cached_tag2;
    }

  };

}

#endif 
    //static shared_ptr<AtomsPackTagObj0> make(const shared_ptr<AtomsPackObj>& x){
    //if(!x->cached_tag0) x->cached_tag0=shared_ptr<AtomsPackTagObj0>(new AtomsPackTagObj0(x));
    //return x->cached_tag0;
    //}

