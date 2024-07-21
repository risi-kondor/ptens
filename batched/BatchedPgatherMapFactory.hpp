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

#ifndef _ptens_BatchedPgatherMapFactory
#define _ptens_BatchedPgatherMapFactory

#include "PtensorMap.hpp"
#include "BatchedPgatherMap.hpp"
#include "LayerMap.hpp"
#include "AtomsPack.hpp"


namespace ptens{


  class BatchedPgatherMapFactory{
  public:


    static BatchedPgatherMap gather_map0(const BatchedLayerMap& map, const BatchedAtomsPack& out, const BatchedAtomsPack& in, 
      const int outk=0, const int ink=0){
      return make(map,*out.obj,*in.obj,outk,ink,0);
    }

    static BatchedPgatherMap gather_map1(const BatchedLayerMap& map, const BatchedAtomsPack& out, const BatchedAtomsPack& in, 
      const int outk=0, const int ink=0){
      return make(map,*out.obj,*in.obj,outk,ink,1);
    }

    static BatchedPgatherMap gather_map2(const BatchedLayerMap& map, const BatchedAtomsPack& out, const BatchedAtomsPack& in, 
      const int outk=0, const int ink=0){
      return make(map,*out.obj,*in.obj,outk,ink,2);
    }


    static shared_ptr<BatchedPgatherMapObj> make(const BatchedLayerMap& map, 
      const BatchedAtomsPackObj& out, const BatchedAtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){
      const int N=map.size();
      PTENS_ASSRT(out.size()==N);
      PTENS_ASSRT(in.size()==N);

      auto out_pack=new BatchedAindexPackB();
      auto in_pack=new BatchedAindexPackB();

      int nrows=0;
      for(int i=0; i<N; i++){
	PgatherMap plan=PgatherMapFactory::make(*map[i].obj,out[i],in[i],outk,ink,gatherk);
	out_pack->push_back(plan.obj->out_map);
	in_pack->push_back(plan.obj->in_map);
	nrows+=plan.obj->in_map->nrows;
      }
      out_pack->nrows=nrows;
      in_pack->nrows=nrows;

      return to_share(new BatchedPgatherMapObj(cnine::to_share(out_pack),cnine::to_share(in_pack)));
    }

  };

}


#endif 
