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

  class AtomsPackTag0;
  class AtomsPackTag1;
  class AtomsPackTag2;

  class AtomsPackTagObj0;
  class AtomsPackTagObj1;
  class AtomsPackTagObj2;


  class AtomsPackTagObj: public cnine::observable<AtomsPackTagObj>{
  public:

    friend class AtomsPackTag0; 
    friend class AtomsPackTag1; 
    friend class AtomsPackTag2; 

    weak_ptr<AtomsPackObj> atoms;

  protected:

    AtomsPackTagObj(const shared_ptr<AtomsPackObj>& x):
      observable(this),
      atoms(x){}
      
  };


  class AtomsPackTagObj0: public AtomsPackTagObj{
  public:

    using AtomsPackTagObj::AtomsPackTagObj;

    const AtomsPackObj0& get_atoms() const{
      return static_cast<AtomsPackObj0&>(*atoms.lock());
    }

  };


  class AtomsPackTagObj1: public AtomsPackTagObj{
  public:

    using AtomsPackTagObj::AtomsPackTagObj;

    const AtomsPackObj1& get_atoms() const{
      return static_cast<AtomsPackObj1&>(*atoms.lock());
    }

  };


  class AtomsPackTagObj2: public AtomsPackTagObj{
  public:

    using AtomsPackTagObj::AtomsPackTagObj;

    const AtomsPackObj2& get_atoms() const{
      return static_cast<AtomsPackObj2&>(*atoms.lock());
    }

  };


  // ---- Tags ------------------------------------------------------------------------------------------------------
  

  class AtomsPackTag0{
  public:

    shared_ptr<AtomsPackTagObj0> obj;

    AtomsPackTag0(){}

    AtomsPackTag0(const AtomsPack& x):
      AtomsPackTag0(x.obj){}

    AtomsPackTag0(const shared_ptr<AtomsPackObj>& x){
      if(!x->cached_tag0) x->cached_tag0=shared_ptr<AtomsPackTagObj0>(new AtomsPackTagObj0(x));
      obj=x->cached_tag0;
    }
    
  };


  class AtomsPackTag1{
  public:

    shared_ptr<AtomsPackTagObj1> obj;

    AtomsPackTag1(){}

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

    AtomsPackTag2(){}

    AtomsPackTag2(const AtomsPack& x):
      AtomsPackTag2(x.obj){}

    AtomsPackTag2(const shared_ptr<AtomsPackObj>& x){
      if(!x->cached_tag2) x->cached_tag2=shared_ptr<AtomsPackTagObj2>(new AtomsPackTagObj2(x));
      obj=x->cached_tag2;
    }

  };

}

#endif 
