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

#ifndef _ptens_RowLevelMap
#define _ptens_RowLevelMap

#include "AtomsPackObj.hpp"
//#include "GatherMapProgram.hpp"
#include "TensorProgram.hpp"
#include "GatherMapB.hpp"
#include "GatherRows.hpp"


namespace ptens{


  class RowLevelMap{
  public:

    typedef cnine::TensorProgram<cnine::GatherRows,cnine::GatherMapB> GatherMapProgram;

    shared_ptr<GatherMapProgram> obj;

    ~RowLevelMap(){}

    RowLevelMap(){};

    RowLevelMap(const GatherMapProgram&& _obj):
      obj(new GatherMapProgram(_obj)){}


    RowLevelMap inv() const{
      return RowLevelMap(obj->inv());
    }

    template<typename TYPE>
    void operator()(cnine::Ltensor<TYPE>& r, const cnine::Ltensor<TYPE>& x){
      (*obj)(r,x);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "RowLevelMap";
    }

    string repr() const{
      return "RowLevelMap";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const RowLevelMap& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
