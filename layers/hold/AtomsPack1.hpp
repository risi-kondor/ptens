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

#ifndef _ptens_AtomsPack1
#define _ptens_AtomsPack1

#include "AtomsPack.hpp"
#include "AtomsPack1obj.hpp"
#include "MessageMap.hpp"

namespace ptens{

  class AtomsPack1{
  public:


    shared_ptr<AtomsPack1obj<int> > obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack1(){}

    AtomsPack1(const AtomsPack& _atoms):
      obj(new AtomsPack1obj<int>(_atoms)){}

    AtomsPack1(shared_ptr<AtomsPack1obj<int> >& _obj):
      obj(_obj){}

    AtomsPack1(const initializer_list<initializer_list<int> >& x):
      obj(new AtomsPack1obj<int>(x)){}

    static AtomsPack1 cat(const vector<AtomsPack1>& list){
      cnine::plist<AtomsPackObjBase*> v;
      for(int i=0; i<list.size()-1; i++)
	v.push_back(list[i+1].obj.get());
      return list[0].obj->cat_maps(v);
    }


  public: // ---- Maps ---------------------------------------------------------------------------------------
    
    
    template<typename SOURCE>
    MessageList overlaps_mlist(const SOURCE& x, const int min_overlaps=1) const{
      return obj->atoms->overlaps_mlist(*x.obj->atoms,min_overlaps);
    }

    template<typename SOURCE>
    MessageMap message_map(const MessageList& list, const SOURCE& source) const{
      return obj->message_map(*list.obj,*source.obj);
    }

    template<typename SOURCE>
    MessageMap overlaps_mmap(const SOURCE& x, const int min_overlaps=1) const{
      return message_map(overlaps_mlist(x,min_overlaps),x);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    int size1() const{
      return obj->size1();
    }

    int size_of(const int i) const{
      return obj->size_of(i);

    }

    int offset(const int i) const{
      return obj->offset(i);
    }

    int index_of(const int i, const int j0) const{
      return obj->index_of(i,j0);
    }

    Atoms operator()(const int i) const{
      return (*obj->atoms)(i);
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
