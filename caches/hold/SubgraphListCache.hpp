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

#ifndef _ptens_SubgraphListCache
#define _ptens_SubgraphListCache

#include "ptr_pair_indexed_object_bank.hpp"
#include "AtomsPackObj.hpp"
#include "GgraphObj.hpp"
#include "SubgraphObj.hpp"


namespace ptens{

  namespace ptens_global{
    extern bool cache_subgraph_lists;
  }

  class SubgraphListCache: 
    public cnine::ptr_pair_indexed_object_bank<GgraphObj,SubgraphObj,shared_ptr<AtomsPackObj> >{
  public:

    typedef cnine::ptr_pair_indexed_object_bank<GgraphObj,SubgraphObj,shared_ptr<AtomsPackObj> > BASE;

    SubgraphListCache():
      BASE([this](const GgraphObj& G, const SubgraphObj& S){
	  return find_subgraphs(G,S);}){}


  public: // ---- Access ------------------------------------------------------------------------------------------


    AtomsPack operator()(const GgraphObj& G, const SubgraphObj& S){
      if(ptens_global::cache_subgraph_lists) return BASE::operator()(G,S); 
      return AtomsPack(find_subgraphs(G,S));
    }


  public: // ---- Find subgraphs ----------------------------------------------------------------------------------


    shared_ptr<AtomsPackObj> find_subgraphs(const GgraphObj& G, const SubgraphObj& S){

      if(S.getn()==1 && S.labeled==false && S.nedges()==0){
	return make_shared<AtomsPackObj>(G.getn());
      }
      
      if(S.getn()==2 && S.labeled==false && S.nedges()==1){
	AtomsPackObj* r=new AtomsPackObj();;
	G.for_each_edge([&](const int i, const int j, const float v){
	    if(i<j) r->push_back({i,j});});
	return cnine::to_share(r);
      }
      
      AtomsPackObj R(cnine::Ltensor<int>(cnine::FindPlantedSubgraphs<float>(G,S)));
      return to_share(new AtomsPackObj(std::move(R)));
    }

  };

}

#endif 
