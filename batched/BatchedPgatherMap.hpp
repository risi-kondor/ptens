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

#ifndef _ptens_BatchedPgatherMap
#define _ptens_BatchedPgatherMap

#include "BatchedPgatherMapObj.hpp"


namespace ptens{

  //class AtomsPackObj;


  class BatchedPgatherMap{
  public:
    
    shared_ptr<BatchedPgatherMapObj> obj;

    BatchedPgatherMap(){
      PTENS_ASSRT(false);}

    BatchedPgatherMap(const shared_ptr<BatchedPgatherMapObj>& x):
      obj(x){}

    const BatchedAindexPackB& in() const{
      return *obj->in_map;
    }

    const BatchedAindexPackB& out() const{
      return *obj->out_map;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedPgatherMap";
    }

    string repr() const{
      return "BatchedPgatherMap";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedPgatherMap& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
