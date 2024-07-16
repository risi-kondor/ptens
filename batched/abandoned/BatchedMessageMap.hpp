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

#ifndef _ptens_BatchedMessageMap
#define _ptens_BatchedMessageMap

#include "AtomsPackObj.hpp"
#include "GatherMapProgramPack.hpp"


namespace ptens{


  class BatchedMessageMap{
  public:

    shared_ptr<cnine::GatherMapProgramPack> obj;

    ~BatchedMessageMap(){}

    BatchedMessageMap(const cnine::GatherMapProgramPack&& _obj):
      obj(new cnine::GatherMapProgramPack(_obj)){}

    template<typename TYPE>
    void operator()(cnine::Ltensor<TYPE>& r, const cnine::Ltensor<TYPE>& x){
      (*obj)(r,x);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedMessageMap";
    }

    string repr() const{
      return "BatchedMessageMap";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedMessageMap& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
