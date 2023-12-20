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

#ifndef _ptens_AtomsPack0
#define _ptens_AtomsPack0

#include "AtomsPack.hpp"
#include "AtomsPack1obj.hpp"
#include "MessageMap.hpp"

namespace ptens{

  class AtomsPack1{
  public:


    shared_ptr<AtomsPack1obj<int> > obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack0(const initializer_list<initializer_list<int> >& x):
      obj(new AtomsPack1obj<int>(x)){}


  public: // ---- Maps ---------------------------------------------------------------------------------------
    
    
    template<typename SOURCE>
    MessageList overlaps_mlist(const SOURCE& x){
      return obj->atoms->overlaps_mlist(*x.obj->atoms);
    }

    template<typename SOURCE>
    MessageMap overlaps_tmap(const SOURCE& x){
      return obj->mmap(obj->atoms->overlaps_mlist(x.obj->atoms));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack1";
    }

    string repr() const{
      return "AtomsPack1";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack1& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
