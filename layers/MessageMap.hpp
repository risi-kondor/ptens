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

#ifndef _ptens_MessageMap
#define _ptens_MessageMap

#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"


namespace ptens{


  class MessageMap{
  public:

    shared_ptr<cnine::GatherMapProgram> obj;

    MessageMap(const cnine::GatherMapProgram&& _obj):
      obj(new cnine::GatherMapProgram(_obj)){}


    MessageMap inv() const{
      return MessageMap(obj->inv());
    }

    template<typename TYPE>
    void operator()(cnine::Ltensor<TYPE>& r, const cnine::Ltensor<TYPE>& x){
      (*obj)(r,x);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "MessageMap";
    }

    string repr() const{
      return "MessageMap";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const MessageMap& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
