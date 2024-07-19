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

#ifndef _ptens_PgatherMapFactory
#define _ptens_PgatherMapFactory

#include "PtensorMap.hpp"
#include "PgatherMap.hpp"
#include "LayerMap.hpp"
#include "AtomsPack.hpp"


namespace ptens{


  class PgatherMapFactory{
  public:


    static PgatherMap gather_map0(const LayerMap& map, const AtomsPack& out, const AtomsPack& in, 
      const int outk=0, const int ink=0){
      return make(*map.obj,*out.obj,*in.obj,outk,ink,0);
    }

    static PgatherMap gather_map1(const LayerMap& map, const AtomsPack& out, const AtomsPack& in, 
      const int outk=0, const int ink=0){
      return make(*map.obj,*out.obj,*in.obj,outk,ink,1);
    }

    static PgatherMap gather_map2(const LayerMap& map, const AtomsPack& out, const AtomsPack& in, 
      const int outk=0, const int ink=0){
      return make(*map.obj,*out.obj,*in.obj,outk,ink,2);
    }


    static shared_ptr<PgatherMapObj> make(const LayerMapObj& map, 
      const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){

      int N=map.tsize();
      int longest=map.max_size();

      auto out_pack=new AindexPackB(N,longest);
      auto in_pack=new AindexPackB(N,longest);

      int c=0;
      int toffset=0;
      map.for_each([&](const int i, const int j){
	  Atoms in_j=in[j];
	  Atoms out_i=out[i];
	  Atoms common=out_i.intersect(in_j);
	  int nix=common.size();
	  
	  if(outk==0) out_pack->set(c,toffset,nix,out.row_offset0(i),out.nrows0(i),out[i](common));
	  if(outk==1) out_pack->set(c,toffset,nix,out.row_offset1(i),out.size_of(i),out[i](common));
	  if(outk==2) out_pack->set(c,toffset,nix,out.row_offset2(i),out.size_of(i),out[i](common));

	  if(ink==0) in_pack->set(c,toffset,nix,in.row_offset0(j),in.nrows0(j),in[j](common));
	  if(ink==1) in_pack->set(c,toffset,nix,in.row_offset1(j),in.size_of(j),in[j](common));
	  if(ink==2) in_pack->set(c,toffset,nix,in.row_offset2(j),in.size_of(j),in[j](common));

	  if(gatherk==0) toffset+=1;
	  if(gatherk==1) toffset+=nix;
	  if(gatherk==2) toffset+=nix*nix;
	  c++;
	});

      out_pack->nrows=toffset;
      in_pack->nrows=toffset;

      auto R=new PgatherMapObj(); 
      R->out_map=to_share(out_pack);
      R->in_map=to_share(in_pack);
      return cnine::to_share(R);
    }
    


  public: // -------------------------------------------------------------------------------------------------


    static PgatherMap gather_map0(const PtensorMap& pmap, const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0){
      return make(*pmap.obj,*out.obj,*in.obj,outk,ink,0);
    }

    static PgatherMap gather_map1(const PtensorMap& pmap, const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0){
      return make(*pmap.obj,*out.obj,*in.obj,outk,ink,0);
    }

    static PgatherMap gather_map2(const PtensorMap& pmap, const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0){
      return make(*pmap.obj,*out.obj,*in.obj,outk,ink,0);
    }


    static shared_ptr<PgatherMapObj> make(const PtensorMapObj& pmap, 
      const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){

      int N=pmap.gmap.tsize();
      int longest=pmap.gmap.max_size();

      auto out_pack=new AindexPackB(N,longest);
      auto in_pack=new AindexPackB(N,longest);

      int c=0;
      int toffset=0;
      pmap.gmap.for_each([&](const int i, const int j){
	  auto common=(*pmap.atoms)[c];
	  int nix=common.size();
	  
	  if(outk==0) out_pack->set(c,toffset,nix,out.row_offset0(i),out.nrows0(i),out[i](common));
	  if(outk==1) out_pack->set(c,toffset,nix,out.row_offset1(i),out.nrows1(i),out[i](common));
	  if(outk==2) out_pack->set(c,toffset,nix,out.row_offset2(i),out.nrows2(i),out[i](common));

	  if(ink==0) in_pack->set(c,toffset,nix,in.row_offset0(j),in.nrows0(j),in[j](common));
	  if(ink==1) in_pack->set(c,toffset,nix,in.row_offset1(j),in.nrows1(j),in[j](common));
	  if(ink==2) in_pack->set(c,toffset,nix,in.row_offset2(j),in.nrows2(j),in[j](common));

	  if(gatherk==0) toffset+=1;
	  if(gatherk==1) toffset+=nix;
	  if(gatherk==2) toffset+=nix*nix;
	  c++;
	});

      out_pack->nrows=toffset;
      in_pack->nrows=toffset;

      auto R=new PgatherMapObj(); 
      R->out_map=to_share(out_pack);
      R->in_map=to_share(in_pack);
      return cnine::to_share(R);
    }

  };




}

#endif 
