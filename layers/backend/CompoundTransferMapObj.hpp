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

#ifndef _ptens_CompundTransferMapObj
#define _ptens_CompundTransferMapOBj

#include "AtomsPackObj.hpp"


namespace ptens{

  class CTMnode{
  public:

    TransferMapObj* tmap;

    ~CTmnode(){
      delete tmap;
    }

  };



  class CompoundTransferMapObj{
  public:

    vector<CTMnode*> nodes;
    vector<LTensor<float>*> vars;

    ~CompoundTransferMapObj(){
      for(auto p:nodes)
	delete p;
      for(auto p:vars)
	delete p;
    }

    CompoundTransferMapObj(const AtomsPackObj& in_atoms, const AtomsPackObj& out_atoms, const TransferMapProgram& prog){
      
    }


    typename<TYPE>
    void operator()(Ltensor<TYPE>& r, const Ltensor<TYPE>& x){
      int nc=r.dim(1);
      
      
      for(auto p:nodes){
	CTMnode& node=*p;
	gather_rows(varsp.out,p.arg
      }
    }
    
  };

}

#endif 
