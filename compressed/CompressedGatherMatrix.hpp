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

#ifndef _ptens_CompressedGatherMatrix
#define _ptens_CompressedGatherMatrix

#include "CompressedGatherMatrixObj.hpp"


namespace ptens{


  class CompressedGatherMatrix{
  public:
    
    shared_ptr<CompressedGatherMatrixObj> obj;

    CompressedGatherMatrix(){
      PTENS_ASSRT(false);}

    CompressedGatherMatrix(const shared_ptr<CompressedGatherMatrixObj>& x):
      obj(x){}


  public: // ---- Access -------------------------------------------------------------------------------------


    void apply(const cnine::TensorView<float>& r, const cnine::TensorView<float>& x) const{
      int nc=x.dims.last();
      obj->apply_to(r.reshape({obj->nrows(),nc}),x.reshape({obj->ncols(),nc}));
    }

    void apply_back(const cnine::TensorView<float>& r, const cnine::TensorView<float>& x) const{
      int nc=r.dims.last();
      obj->apply_transp_to(r.reshape({obj->ncols(),nc}),x.reshape({obj->nrows(),nc}));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "CompressedGatherMatrix";
    }

    string repr() const{
      return "CompressedGatherMatrix";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const CompressedGatherMatrix& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
