/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _ptens_AtomsPack0batch
#define _ptens_AtomsPack0batch

#include "AtomsPack.hpp"
#include "AtomsPack0obj.hpp"
#include "AtomsPack1obj.hpp"
#include "AtomsPack2obj.hpp"
#include "MessageMap.hpp"

namespace ptens{

  class AtomsPack0batch{
  public:


    shared_ptr<AtomsPack0batchObj<int> > obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack0batch(){}

    AtomsPack0batch(const int n):
      obj(new AtomsPack0batchobj<int>(n)){}

    AtomsPack0batch(const AtomsPack& _atoms):
      obj(new AtomsPack0batchobj<int>(_atoms)){}

    AtomsPack0batch(shared_ptr<AtomsPack0batchobj<int> >& _obj):
      obj(_obj){}

    AtomsPack0batch(const initializer_list<initializer_list<int> >& x):
      obj(new AtomsPack0batchobj<int>(x)){}

    static AtomsPack0batch cat(const vector<AtomsPack0batch>& list){
      cnine::plist<AtomsPackObjBase*> v;
      for(int i=0; i<list.size()-1; i++)
	v.push_back(list[i+1].obj.get());
      return list[0].obj->cat_maps(v);
    }


  public: // ---- Maps ---------------------------------------------------------------------------------------
    
    
    template<typename SOURCE>
    MessageListBatch overlaps_mlist(const SOURCE& x) const{
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

    //Atoms operator()(const int i) const{
    //return (*obj->atoms)(i);
    //}

    vector<vector<int> > as_vecs() const{
      return obj->atoms->as_vecs();
    }

    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack0batch";
    }

    string repr() const{
      return "AtomsPack0batch";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack0batch& v){
      stream<<v.str(); return stream;}

  };


}

#endif 


// AtomsPackObj <- MessageList <- MessageListObj
