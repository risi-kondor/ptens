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

#ifndef _ptens_CompoundTransferMapObj
#define _ptens_CompoundTransferMapObj

#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"


namespace ptens{



  class CompoundTransferMapObj{
  public:

    cnine::GatherMapProgram prog;

    CompoundTransferMapObj(const int in_dim, const int out_dim, const int column_multiplier=1):
      prog(cnine::dims(in_dim,1),cnine::dims(out_dim,column_multiplier)){}

    CompoundTransferMapObj(const int in_dim, const int out_dim, const cnine::GatherMapB& g):
      prog(cnine::dims(in_dim,1),cnine::dims(out_dim,1),g){}

    ~CompoundTransferMapObj(){
    }

  };

}

#endif 
  /*
  class CTMnode{
  public:

    TransferMapObj* tmap;

    ~CTmnode(){
      delete tmap;
    }

  };
  */
