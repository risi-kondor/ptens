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

#ifndef _ptens_BatchedGatherPlanFactory
#define _ptens_BatchedGatherPlanFactory

//#include "PtensorMap.hpp"
#include "BatchedGatherPlan.hpp"
#include "BatchedLayerMap.hpp"
#include "LayerMap.hpp"
#include "AtomsPack.hpp"


namespace ptens{


  class BatchedGatherPlanFactory{
  public:


    template<int aoutk, int aink>
    static BatchedGatherPlan gather_map0(const BatchedLayerMap& map, const BatchedAtomsPack<aoutk>& out, 
      const BatchedAtomsPack<aink>& in, const int outk=-1, const int ink=-1){
      if(outk>0) PTENS_ASSRT(outk==aoutk);
      if(ink>0) PTENS_ASSRT(ink==aink);
      return make_or_cached<aoutk,aink>(map,*out.obj,*in.obj,0);
    }

    template<int aoutk, int aink>
    static BatchedGatherPlan gather_map1(const BatchedLayerMap& map, const BatchedAtomsPack<aoutk>& out, 
      const BatchedAtomsPack<aink>& in, const int outk=-1, const int ink=-1){
      if(outk>0) PTENS_ASSRT(outk==aoutk);
      if(ink>0) PTENS_ASSRT(ink==aink);
      return make_or_cached<aoutk,aink>(map,*out.obj,*in.obj,1);
    }

    template<int aoutk, int aink>
    static BatchedGatherPlan gather_map2(const BatchedLayerMap& map, const BatchedAtomsPack<aoutk>& out, 
      const BatchedAtomsPack<aink>& in, const int outk=0, const int ink=0){
      if(outk>0) PTENS_ASSRT(outk==aoutk);
      if(ink>0) PTENS_ASSRT(ink==aink);
      return make_or_cached<aoutk,aink>(map,*out.obj,*in.obj,2);
    }

    template<int outk, int ink>
    static shared_ptr<BatchedGatherPlanObj> make_or_cached(const BatchedLayerMap& map, 
      const BatchedAtomsPackObj& out, const BatchedAtomsPackObj& in, const int gatherk=0){

      if(ptens_global::batched_gather_plan_cache.contains(*map.obj,out,in,9*outk+3*ink+gatherk))
	return ptens_global::batched_gather_plan_cache(*map.obj,out,in,9*outk+3*ink+gatherk);

      auto R=make<outk,ink>(map,out,in,gatherk);
      ptens_global::batched_gather_plan_cache.insert(*map.obj,out,in,9*outk+3*ink+gatherk,R);
      return R;
    }

    template<int outk, int ink>
    static shared_ptr<BatchedGatherPlanObj> make(const BatchedLayerMap& map, 
      const BatchedAtomsPackObj& out, const BatchedAtomsPackObj& in, const int gatherk=0){
      const int N=map.size();
      PTENS_ASSRT(out.size()==N);
      PTENS_ASSRT(in.size()==N);

      auto out_pack=new BatchedAindexPackB();
      auto in_pack=new BatchedAindexPackB();

      int nrows=0;
      for(int i=0; i<N; i++){
	GatherPlan plan=GatherPlanFactory::make_or_cached(*map[i].obj,out[i],in[i],outk,ink,gatherk);
	out_pack->push_back(plan.obj->out_map);
	in_pack->push_back(plan.obj->in_map);
	nrows+=plan.obj->in_map->nrows;
      }
      out_pack->nrows=nrows;
      in_pack->nrows=nrows;

      return to_share(new BatchedGatherPlanObj(cnine::to_share(out_pack),cnine::to_share(in_pack)));
    }

  };

}


#endif 
    /*
    template<int outk, int ink>
    static BatchedGatherPlan gather_map0(const BatchedLayerMap& map, const BatchedAtomsPack<outk>& out, 
      const BatchedAtomsPack<ink>& in){
      return make<outk,ink>(map,*out.obj,*in.obj,0);
    }

    template<int outk, int ink>
    static BatchedGatherPlan gather_map1(const BatchedLayerMap& map, const BatchedAtomsPack<outk>& out, 
      const BatchedAtomsPack<ink>& in){
      return make<outk,ink>(map,*out.obj,*in.obj,1);
    }

    template<int outk, int ink>
    static BatchedGatherPlan gather_map2(const BatchedLayerMap& map, const BatchedAtomsPack<outk>& out, 
      const BatchedAtomsPack<ink>& in){
      return make<outk,ink>(map,*out.obj,*in.obj,2);
    }
    */

