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

#ifndef _Ptens_GraphBatchObj
#define _Ptens_GraphBatchObj

#include "object_pack_s.hpp"
#include "GgraphObj.hpp"


namespace ptens{


  class GgraphBatchObj: object_pack_s<GgraphObj>{
  public:


  public: //  ---- Named constructors -------------------------------------------------------------------------



  public: // ---- Conversions ---------------------------------------------------------------------------------


  public: // ---- Access --------------------------------------------------------------------------------------


  public: // ---- Operations ---------------------------------------------------------------------------------


    GgraphBatchObj permute(const cnine::permutation pi) const{
      return mapcar<GgraphBatchObj,GgraphObj>([&](const GgrpaphObj& x){return x.permute(pi);});
    }

    AtomsPackBatch subgraphs(const SubgraphObj& H){
      return mapcar<AtomsPackBatch,AtomsPack>([&](const GgrpaphObj& x){return x.subgraphs(H);});
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------



  };

}

#endif 


      //GgraphBatchObj R;
      //for(auto p: obj)
      //R.push_back(cnine::to_share(new GgraphObj(obj->permute(pi))));
      //return R;
