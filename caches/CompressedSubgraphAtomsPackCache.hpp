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

#ifndef _ptens_CompressedSubgraphAtomsPackCache
#define _ptens_CompressedSubgraphAtomsPackCache

#include "ptr_arg_indexed_cache.hpp"
#include "CompressedAtomsPack.hpp"
#include "GgraphObj.hpp"
#include "SubgraphObj.hpp"


namespace ptens{

  class CompressedSubgraphAtomsPackCache{
  public:

    cnine::ptr_arg_indexed_cache<AtomsPackObj,int,shared_ptr<CompressedAtomsPackObj> > cache;

    CompressedSubgraphAtomsPackCache(){}

    CompressedAtomsPack operator()(const GgraphObj& G, const shared_ptr<SubgraphObj>& S, const int nvecs){
      AtomsPack A=G.subgraphs(S);
      if(cache.contains(*A.obj,nvecs)) return cache(*A.obj,nvecs);
      auto r=cnine::to_share(new CompressedAtomsPackObj(A.obj,S->evecs.cols(0,nvecs).transp()));
      cache.insert(*A.obj,nvecs,r);
      return r;
    }

  };

}

#endif 

