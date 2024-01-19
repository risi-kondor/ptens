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
 */


#ifndef _ptens_BatchedAtomsPack
#define _ptens_BatchedAtomsPack

#include "BatchedAtomsPackObj.hpp"


namespace ptens{

  class BatchedAtomsPack{
  public:


    shared_ptr<BatchedAtomsPackObj> obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedAtomsPack():
      obj(new BatchedAtomsPackObj()){}

    BatchedAtomsPack(BatchedAtomsPackObj* _obj):
      obj(_obj){}

    BatchedAtomsPack(shared_ptr<BatchedAtomsPackObj> _obj):
      obj(_obj){}


  public: // ----- Access ------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    AtomsPack operator[](const int i) const{
      PTENS_ASSRT(i<size());
      return obj->obj[i];
    }
    

  public: // ---- Operations ---------------------------------------------------------------------------------


    //BatchedAtomsPack permute(const cnine::permutation& pi){
    //return BatchedAtomsPack(new BatchedAtomsPackObj(obj->permute(pi)));
    //} 
    
    //MessageListBatch overlaps_mlist(const BatchedAtomsPack& y){
    //return obj->overlaps_mlist(*y.obj);
    //}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedAtomsPack";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedAtomsPack& v){
      stream<<v.str(); return stream;}

  };


}

#endif 
