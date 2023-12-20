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

    //MessageMap(const int in_dim, const int out_dim, const cnine::GatherMapB& g):
    //obj(new cnine::GatherMapB(g,in_dim,out_dim)){}

  };

}

#endif 
