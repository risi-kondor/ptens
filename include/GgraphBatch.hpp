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

#ifndef _Ptens_GraphBatch
#define _Ptens_GraphBatch

#include "GgraphBatchObj.hpp"


namespace ptens{


  class GgraphBatch{
  public:

    typedef GgraphBatchObj BASE;

    shared_ptr<BASE> obj;


  public: //  ---- Named constructors -------------------------------------------------------------------------


  public: // ---- Access --------------------------------------------------------------------------------------


  public: // ---- Operations ----------------------------------------------------------------------------------


    GgraphBatch permute(const cnine::permutation& pi) const{
      return GgraphBatch(new BASE(obj->permute(pi)));
    }

    AtomsPackBatch subgraphs(const Subgraph& H) const{
      return obj->subgraphs(*H.obj);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "ptens::GgraphBatch";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const GgraphBatch& x){
      stream<<x.str(); return stream;}

  }

}

#endif 
