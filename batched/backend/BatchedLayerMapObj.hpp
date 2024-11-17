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

#ifndef _ptens_BatchedLayerMapObj
#define _ptens_BatchedLayerMapObj

#include "observable.hpp"
//#include "LayerMap.hpp"


namespace ptens{


  class BatchedLayerMapObj: public cnine::observable<BatchedLayerMapObj>{
  public:

    vector<shared_ptr<LayerMapObj> > maps;

    BatchedLayerMapObj():
      observable(this){}


    static shared_ptr<BatchedLayerMapObj> overlaps_map(const BatchedAtomsPackObj& out, const BatchedAtomsPackObj& in, 
      const int min_overlaps=1){

      PTENS_ASSRT(out.size()==in.size());
      int N=out.size();
      auto R=new BatchedLayerMapObj();
      for(int i=0; i<N; i++){
	auto& out_atoms=out[i];
	auto& in_atoms=in[i];
	if(ptens_global::overlaps_maps_cache.contains(out_atoms,in_atoms))
	  R->maps.push_back(ptens_global::overlaps_maps_cache(out_atoms,in_atoms));
	else{
	  auto r=LayerMapObj::overlaps_map(out_atoms,in_atoms,min_overlaps);
	  ptens_global::overlaps_maps_cache.insert(out_atoms,in_atoms,r);
	  R->maps.push_back(r);
	}
      }

      return cnine::to_share(R);
    }

  };

}

#endif 
