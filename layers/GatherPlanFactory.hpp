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

#ifndef _ptens_GatherPlanFactory
#define _ptens_GatherPlanFactory

#include "GatherPlan.hpp"
#include "LayerMap.hpp"
#include "AtomsPack.hpp"
#include "ptr_triple_indexed_cache.hpp"


namespace ptens{


  class GatherPlanFactory{
  public:


    static GatherPlan gather_map0(const LayerMap& map, const AtomsPack& out, const AtomsPack& in, 
      const int outk=0, const int ink=0){
      return make_or_cached(*map.obj,*out.obj,*in.obj,outk,ink,0);
    }

    static GatherPlan gather_map1(const LayerMap& map, const AtomsPack& out, const AtomsPack& in, 
      const int outk=0, const int ink=0){
      return make_or_cached(*map.obj,*out.obj,*in.obj,outk,ink,1);
    }

    static GatherPlan gather_map2(const LayerMap& map, const AtomsPack& out, const AtomsPack& in, 
      const int outk=0, const int ink=0){
      return make_or_cached(*map.obj,*out.obj,*in.obj,outk,ink,2);
    }

    static shared_ptr<GatherPlanObj> make_or_cached(const LayerMapObj& map, 
      const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){

      if(ptens_global::gather_plan_cache.contains(map,out,in,9*outk+3*ink+gatherk))
	return ptens_global::gather_plan_cache(map,out,in,9*outk+3*ink+gatherk);

      auto R=make(map,out,in,outk,ink,gatherk);
      ptens_global::gather_plan_cache.insert(map,out,in,9*outk+3*ink+gatherk,R);
      return R;
    }


    static shared_ptr<GatherPlanObj> make(const LayerMapObj& map, 
      const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){

      int N=map.tsize();
      int longest=map.max_size();

      cnine::map_of_lists<int,int> out_lists;
      cnine::map_of_lists<int,int> in_lists;

      auto out_pack=new AindexPackB(N,longest);
      auto in_pack=new AindexPackB(N,longest);

      int c=0;
      int toffset=0;
      map.for_each([&](const int i, const int j){
	  Atoms in_j=in[j];
	  Atoms out_i=out[i];
	  //Atoms common=out_i.intersect(in_j);
	  //int nix=common.size();
	  auto[out_ix,in_ix]=out_i.intersecting(in_j);
	  int nix=out_ix.size();

	  //if(outk==0) out_pack->set(c,toffset,nix,out.row_offset0(i),out.nrows0(i),out[i](common)); // is this correct?
	  //if(outk==1) out_pack->set(c,toffset,nix,out.row_offset1(i),out.size_of(i),out[i](common));
	  //if(outk==2) out_pack->set(c,toffset,nix,out.row_offset2(i),out.size_of(i),out[i](common));

	  //if(ink==0) in_pack->set(c,toffset,nix,in.row_offset0(j),in.nrows0(j),in[j](common));
	  //if(ink==1) in_pack->set(c,toffset,nix,in.row_offset1(j),in.size_of(j),in[j](common));
	  //if(ink==2) in_pack->set(c,toffset,nix,in.row_offset2(j),in.size_of(j),in[j](common));

	  if(outk==0) out_pack->set(c,toffset,nix,out.row_offset0(i),out.nrows0(i),out_ix); // is this correct?
	  if(outk==1) out_pack->set(c,toffset,nix,out.row_offset1(i),out.size_of(i),out_ix);
	  if(outk==2) out_pack->set(c,toffset,nix,out.row_offset2(i),out.size_of(i),out_ix);

	  if(ink==0) in_pack->set(c,toffset,nix,in.row_offset0(j),in.nrows0(j),in_ix);
	  if(ink==1) in_pack->set(c,toffset,nix,in.row_offset1(j),in.size_of(j),in_ix);
	  if(ink==2) in_pack->set(c,toffset,nix,in.row_offset2(j),in.size_of(j),in_ix);

	  out_lists.push_back((*out_pack)(c,2),c);
	  in_lists.push_back((*in_pack)(c,2),c);

	  if(gatherk==0) toffset+=1;
	  if(gatherk==1) toffset+=nix;
	  if(gatherk==2) toffset+=nix*nix;
	  c++;
	});

      out_pack->gather_map=cnine::GatherMapB(out_lists);
      in_pack->gather_map=cnine::GatherMapB(in_lists);

      out_pack->nrows=toffset;
      out_pack->n_gather_lists=out_lists.size();
      if(outk==0) out_pack->n_input_rows=out.nrows0();
      if(outk==1) out_pack->n_input_rows=out.nrows1();
      if(outk==2) out_pack->n_input_rows=out.nrows2();

      in_pack->nrows=toffset;
      in_pack->n_gather_lists=in_lists.size();
      if(ink==0) in_pack->n_input_rows=in.nrows0();
      if(ink==1) in_pack->n_input_rows=in.nrows1();
      if(ink==2) in_pack->n_input_rows=in.nrows2();

      auto R=new GatherPlanObj(); 
      R->out_map=to_share(out_pack);
      R->in_map=to_share(in_pack);

      auto r=cnine::to_share(R);
      out.related_gatherplans.emplace_back(r);
      return r;
    }
    
  };

}

#endif 



