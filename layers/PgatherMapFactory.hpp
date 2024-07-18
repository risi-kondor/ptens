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
#include "AtomsPack.hpp"


namespace ptens{

  //class AtomsPackObj;


  class PgatherMapFactory{
  public:

    static PgatherMap gather_map0(const PtensorMap& pmap, const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0){
      return make(*pmap.obj,*out.obj,*in.obj,outk,ink,0);
    }

    static PgatherMap gather_map1(const PtensorMap& pmap, const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0){
      return make(*pmap.obj,*out.obj,*in.obj,outk,ink,0);
    }

    static PgatherMap gather_map2(const PtensorMap& pmap, const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0){
      return make(*pmap.obj,*out.obj,*in.obj,outk,ink,0);
    }


  public: // -------------------------------------------------------------------------------------------------


    static shared_ptr<PgatherMapObj> make(const PtensorMapObj& pmap, 
      const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){

      int N=pmap.gmap.size();
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

      auto R=new PgatherMapObj(); 
      R->out_map=to_share(out_pack);
      R->in_map=to_share(in_pack);
      return cnine::to_share(R);
    }

  };


}

#endif 
