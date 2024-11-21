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

#ifndef _ptens_BatchedLayerMap
#define _ptens_BatchedLayerMap

#include "object_pack_s.hpp"
//#include "LayerMap.hpp"
#include "BatchedLayerMapObj.hpp"
#include "BatchedAtomsPack.hpp"
//#include "BatchedAindexPack.hpp"

namespace ptens{


  class BatchedLayerMap{ 
  public:

    //vector<shared_ptr<LayerMapObj> > maps;
    shared_ptr<BatchedLayerMapObj> obj;

    BatchedLayerMap(const shared_ptr<BatchedLayerMapObj>& _obj):
      obj(_obj){}


  public: // ---- Named constructors ------------------------------------------------------------------------


    static BatchedLayerMap overlaps_map(const BatchedAtomsPackBase& out, const BatchedAtomsPackBase& in, 
      const int min_overlaps=1){

      if(ptens_global::batched_overlaps_maps_cache.contains(*out.obj,*in.obj))
	return (ptens_global::batched_overlaps_maps_cache(*out.obj,*in.obj));

      auto r=BatchedLayerMapObj::overlaps_map(*out.obj,*in.obj,min_overlaps);
      ptens_global::batched_overlaps_maps_cache.insert(*out.obj,*in.obj,r);
      return BatchedLayerMap(r);

      //PTENS_ASSRT(out.size()==in.size());
      //int N=out.size();
      //BatchedLayerMap R;
      //for(int i=0; i<N; i++)
      //R.maps.push_back(LayerMapObj::overlaps_map(*out[i].obj,*in[i].obj,min_overlaps));
      //return R;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      //return maps.size();
      return obj->maps.size();
    }

    LayerMap operator[](const int i) const{
      PTENS_ASSRT(i<obj->maps.size());
      return obj->maps[i];
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedLayerMap";
    }

    string repr() const{
      return "BatchedLayerMap";
    }

    string str(const string indent="") const{
      return "";
    }

    friend ostream& operator<<(ostream& stream, const BatchedLayerMap& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
