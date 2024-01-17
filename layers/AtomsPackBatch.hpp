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


#ifndef _ptens_AtomsPackBatch
#define _ptens_AtomsPackBatch

#include "AtomsPackBatchObj.hpp"


namespace ptens{

  class AtomsPackBatch{
  public:


    shared_ptr<AtomsPackBatchObj> obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackBatch():
      obj(new AtomsPackBatchObj()){}

    AtomsPackBatch(AtomsPackBatchObj* _obj):
      obj(_obj){}

    AtomsPackBatch(shared_ptr<AtomsPackBatchObj> _obj):
      obj(_obj){}


  public: // ----- Access ------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    AtomsPack operator[](const int i){
      PTENS_ASSRT(i<size());
      return obj->obj[i];
    }
    

  public: // ---- Operations ---------------------------------------------------------------------------------


    AtomsPackBatch permute(const cnine::permutation& pi){
      return AtomsPackBatch(new AtomsPackBatchObj(obj->permute(pi)));
    } 
    
    MessageList overlaps_mlist(const AtomsPackBatch& y){
      return obj->overlaps_mlist(*y.obj);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPackBatch";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPackBatch& v){
      stream<<v.str(); return stream;}

  };


}

#endif 
