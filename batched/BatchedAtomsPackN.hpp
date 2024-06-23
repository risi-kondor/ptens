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
#include "AtomsPackObjBase.hpp"
#include "AtomsPack1obj.hpp"


namespace ptens{

  template<typename SUB>
  class BatchedAtomsPackN{
  public:


    shared_ptr<BatchedAtomsPackNobj<SUB> > obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedAtomsPackN(){}

    BatchedAtomsPackN(BatchedAtomsPackNobj<SUB>* _obj):
      obj(_obj){}

    BatchedAtomsPackN(BatchedAtomsPackNobj<SUB>&& _obj):
      obj(new BatchedAtomsPackNobj<SUB>(_obj)){}

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
    */

    static BatchedAtomsPackN cat(const vector<BatchedAtomsPackN>& list){
      auto R=new BatchedAtomsPackNobj<SUB>();
      PTENS_ASSRT(list.size()>0);
      int N=list[0].size();
      for(int i=0; i<N; i++){
	cnine::plist<AtomsPackObjBase*> v;
	for(int j=0; j<list.size()-1; j++)
	  v.push_back(list[j+1].obj->obj[i].get());
	//v.push_back(static_cast<AtomsPackObjBase*>(const_cast<SUB*>(&list[j+1][i])));
	//v.push_back(static_cast<AtomsPackObjBase*>(const_cast<AtomsPack1obj<int>*>(&list[j+1][i])));
        //R->obj.push_back(const_cast<SUB&>(list[0][i]).cat_maps(v));
        R->obj.push_back(list[0].obj->obj[i]->cat_maps(v));
      }
      R->make_row_offsets();
      return BatchedAtomsPackN(R);
    }


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
