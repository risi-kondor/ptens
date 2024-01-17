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
#include "AtomsPack0obj.hpp"
#include "AtomsPack1obj.hpp"
#include "AtomsPack2obj.hpp"
#include "MessageMap.hpp"

namespace ptens{

  class AtomsPack0{
  public:


    shared_ptr<AtomsPack0obj<int> > obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack0(){}

    AtomsPack0(const int n):
      obj(new AtomsPack0obj<int>(n)){}

    AtomsPack0(const AtomsPack& _atoms):
      obj(new AtomsPack0obj<int>(_atoms)){}

    AtomsPack0(shared_ptr<AtomsPack0obj<int> >& _obj):
      obj(_obj){}

    AtomsPack0(const initializer_list<initializer_list<int> >& x):
      obj(new AtomsPack0obj<int>(x)){}

    static AtomsPack0 cat(const vector<AtomsPack0>& list){
      cnine::plist<AtomsPackObjBase*> v;
      for(int i=0; i<list.size()-1; i++)
	v.push_back(list[i+1].obj.get());
      return list[0].obj->cat_maps(v);
    }


  public: // ---- Maps ---------------------------------------------------------------------------------------
    
    
    template<typename SOURCE>
    MessageList overlaps_mlist(const SOURCE& x) const{
      return obj->atoms->overlaps_mlist(*x.obj->atoms);
    }

    template<typename SOURCE>
    MessageMap message_map(const MessageList& list, const SOURCE& source) const{
      return obj->message_map(*list.obj,*source.obj);
    }

    template<typename SOURCE>
    MessageMap overlaps_mmap(const SOURCE& x) const{
      return message_map(overlaps_mlist(x),x);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    int offset(const int i) const{
      return obj->offset(i);
    }

    Atoms operator()(const int i) const{
      return (*obj->atoms)(i);
    }

    vector<vector<int> > as_vecs() const{
      return obj->atoms->as_vecs();
    }

    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack0";
    }

    string repr() const{
      return "AtomsPack0";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack0& v){
      stream<<v.str(); return stream;}

  };


  class AtomsPack0pack: public vector<AtomsPack0>{
  public:

    int size0() const{
      int t=0;
      for(auto& p:*this)
	t+=p.size();
      return t;
    }

  };

}

#endif 


// AtomsPackObj <- MessageList <- MessageListObj
