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

#ifndef _Ptens_BatchedGgraphObj
#define _Ptens_BatchedGgraphObj

#include "object_pack_s.hpp"
#include "Ggraph.hpp"
#include "BatchedAtomsPack.hpp"


namespace ptens{


  class BatchedGgraphObj: public cnine::object_pack_s<GgraphObj>{
  public:

    typedef cnine::object_pack_s<GgraphObj> BASE;

    using BASE::BASE;


    BatchedGgraphObj(const vector<int>& keys){
      for(auto p: keys)
	obj.push_back(ptens_global::graph_cache(p));
    }


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static BatchedGgraphObj* from_edge_list_p(const vector<int>& sizes, const cnine::Tensor<int>& M){//, const bool cached=false){
      PTENS_ASSRT(sizes.size()>0);
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.dim(0)==2);
      auto R=new BatchedGgraphObj();

      int j=0;
      int t=0;
      int upper=sizes[0];
      for(int i=0; i<sizes.size(); i++){
	while(M(j,0)<upper){
	  PTENS_ASSRT(j<M.dim(1));
	  PTENS_ASSRT(M(j,1)<upper);
	  j++;
	}
	//if(cached){
	//auto G=Ggraph::cached_from_edge_list(M.cols(t,j-t));
	//R->obj.push_back(G.obj);
	//}else{
	  R->obj.push_back(to_share(GgraphObj::from_edges_p(M.cols(t,j-t))));
	  //}
	t=j;
	upper+=sizes[i];
      }
      //if(cached){
      //auto G=Ggraph::cached_from_edge_list(M.cols(t,M.dim(1)-t));
      //R->obj.push_back(G.obj);
      //}else{
	R->obj.push_back(to_share(GgraphObj::from_edges_p(M.cols(t,M.dim(1)-t))));
	//R.obj.push_back(new GgraphObj(M.cols(t,M.dim(1)-t)));
	//}
      return R;
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


  public: // ---- Access --------------------------------------------------------------------------------------


  public: // ---- Operations ---------------------------------------------------------------------------------


    BatchedGgraphObj permute(const cnine::permutation pi){
      return mapcar<BatchedGgraphObj,GgraphObj>([&](const GgraphObj& x){return x.permute(pi);});
    }

    /*
    template<int k>
    BatchedAtomsPack<k> subgraphs(const shared_ptr<SubgraphObj>& H){
      auto R=new BatchedAtomsPackObj();
      for(auto& p:obj)
	R->push_back(p->subgraphs(H).obj);
      return R;
    }
    */

    BatchedAtomsPackBase subgraphs(const shared_ptr<SubgraphObj>& H){
      auto R=new BatchedAtomsPackObj();
      for(auto& p:obj)
	R->push_back(p->subgraphs(H).obj);
      return R;
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------



  };

}

#endif 

