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

#ifndef _ptens_BatchedAtomsPackN
#define _ptens_BatchedAtomsPackN

#include "BatchedAtomsPackNobj.hpp"


namespace ptens{

  template<typename SUB>
  class BatchedAtomsPackN{
  public:


    shared_ptr<BatchedAtomsPackNobj<SUB> > obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedAtomsPackN(){}

    BatchedAtomsPackN(const vector<shared_ptr<SUB> >& x):
      obj(new BatchedAtomsPackNobj<SUB>(x)){}

    BatchedAtomsPackN(const BatchedAtomsPack& x):
      obj(new BatchedAtomsPackNobj<SUB>(*x.obj)){}


    /*
    BatchedAtomsPackN(){}

    BatchedAtomsPackN(const int n):
      obj(new BatchedAtomsPackNobj<int>(n)){}

    BatchedAtomsPackN(const AtomsPack& _atoms):
      obj(new BatchedAtomsPackNobj<int>(_atoms)){}

    BatchedAtomsPackN(shared_ptr<BatchedAtomsPackNobj<int> >& _obj):
      obj(_obj){}

    BatchedAtomsPackN(const initializer_list<initializer_list<int> >& x):
      obj(new BatchedAtomsPackNobj<int>(x)){}

    static BatchedAtomsPackN cat(const vector<BatchedAtomsPackN>& list){
      cnine::plist<AtomsPackObjBase*> v;
      for(int i=0; i<list.size()-1; i++)
	v.push_back(list[i+1].obj.get());
      return list[0].obj->cat_maps(v);
    }
    */

  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    int tsize() const{
      return obj->tsize();
    }

    int offset(const int i) const{
      return obj->offset(i);
    }

    int nrows(const int i) const{
      return obj->nrows(i);
    }

    SUB& operator[](const int i){
      return (*obj)[i];
    }

    const SUB& operator[](const int i) const{
      return (*obj)[i];
    }

    //vector<vector<int> > as_vecs() const{
    //return obj->atoms->as_vecs();
    //}

    

  public: // ---- Access ------------------------------------------------------------------------------------




  public: // ---- Maps ---------------------------------------------------------------------------------------
    

    // unused
    /*
    template<typename SOURCE>
    BatchedMessageList overlaps_mlist(const SOURCE& x) const{
      return obj->overlaps_mlist(*x.obj);
    }

    template<typename SOURCE>
    BatchedMessageMap message_map(const BatchedMessageList& list, const SOURCE& source) const{
      return obj->message_map(list.obj,*source.obj);
    }

    template<typename SOURCE>
    BatchedMessageMap inverse_message_map(const BatchedMessageList& list, const SOURCE& source) const{
      return obj->inverse_message_map(list.obj,*source.obj);
    }

    template<typename SOURCE>
    BatchedMessageMap overlaps_mmap(const SOURCE& x) const{
      return message_map(overlaps_mlist(x),x);
    }

    template<typename SOURCE>
    BatchedMessageMap inverse_overlaps_mmap(const SOURCE& x) const{
      return inverse_message_map(overlaps_mlist(x),x);
    }
    */

 
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedAtomsPackN";
    }

    string repr() const{
      return "BatchedAtomsPackN";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedAtomsPackN& v){
      stream<<v.str(); return stream;}

  };


}

#endif 


// AtomsPackObj <- MessageList <- MessageListObj
