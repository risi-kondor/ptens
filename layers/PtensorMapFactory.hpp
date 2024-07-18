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

#ifndef _ptens_PtensorMapFactory
#define _ptens_PtensorMapFactory

#include "PtensorMapObj.hpp"
#include "PtensorMap.hpp"
#include "AtomsPack.hpp"


namespace ptens{

  //class AtomsPackObj;


  class PtensorMapFactory{
  public:

    static PtensorMap overlaps(const AtomsPack& out, const AtomsPack& in, const int outk=0, const int ink=0, const int gatherk=0){
      return overlaps(*out.obj,*in.obj,outk,ink,gatherk);
    }

    static PtensorMap overlaps(const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){
      auto g=make_overlaps(out,in);
      return PtensorMap(make_intersects(g,out,in,outk,ink,gatherk));
    }

    static shared_ptr<PtensorMapObj> overlaps_obj(const AtomsPackObj& out, const AtomsPackObj& in, const int outk=0, const int ink=0, const int gatherk=0){
      auto g=make_overlaps(out,in);
      return make_intersects(g,out,in,outk,ink,gatherk);
    }


    // ------------------------------------------------------------------------------------------------------


    static cnine::map_of_lists<int,int> make_overlaps(const AtomsPackObj& out_atoms, const AtomsPackObj& in_atoms){
      //cnine::flog timer("PtensorMapFactory::make_overlaps");
      cnine::map_of_lists<int,int> gmap;

      if(in_atoms.size()<10){
	for(int i=0; i<out_atoms.size(); i++){
	  auto v=(out_atoms)(i);
	  for(int j=0; j<in_atoms.size(); j++){
	    auto w=(in_atoms)(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      gmap.push_back(i,j);
	  }
	}
      }else{
	unordered_map<int,vector<int> > map;
	for(int j=0; j<in_atoms.size(); j++){
	  auto w=(in_atoms)(j);             
	  for(auto p:w){
	    auto it=map.find(p);
	    if(it==map.end()) map[p]=vector<int>({j});
	    else it->second.push_back(j);
	  }
	}          
	for(int i=0; i<out_atoms.size(); i++){
	  auto v=(out_atoms)(i);
	  for(auto p:v){
	    auto it=map.find(p);
	    if(it!=map.end())
	      for(auto q:it->second)
		gmap.push_back(i,q);
	  }
	}
      }
      return gmap;
    }
    
    static shared_ptr<PtensorMapObj> make_intersects(const cnine::map_of_lists<int,int>& gmap, 
      const AtomsPackObj& out_atoms, const AtomsPackObj& in_atoms, const int outk, const int ink, const int gatherk){

      cnine::ftimer timer("PtensorMapFactory::make_intersects");
      auto R=new PtensorMapObj();
      R->n_in=in_atoms.size();
      R->n_out=out_atoms.size();
      //outk=out_atoms.getk();
      //ink=in_atoms.getk();

      vector<vector<int> > intersections;
      size_t max_intersects=0;

      gmap.for_each([&](const int i, const int j){
	  Atoms in_j=(in_atoms)[j];
	  Atoms out_i=(out_atoms)[i];
	  Atoms common=out_i.intersect(in_j);
	  intersections.push_back(common);
	  cnine::bump(max_intersects,common.size());
	  auto _in=in_j(common);
	  auto _out=out_i(common);
	  R->atoms->push_back(common);
	  R->in->push_back(j,_in);
	  R->out->push_back(i,_out);
	  R->in->count1+=_in.size();
	  R->in->count2+=_in.size()*_in.size();
	  R->out->count1+=_out.size();
	  R->out->count2+=_out.size()*_out.size();
	});

      return cnine::to_share(R);
    }


  };

}


#endif 


      /*
      auto out_pack=new AindexPackB(intersections.size(),max_intersects);
      auto in_pack=new AindexPackB(intersections.size(),max_intersects);

      int c=0;
      int toffset=0;
      gmap.for_each([&](const int i, const int j){
	  auto& common=intersections[c];
	  int nix=common.size();
	  
	  if(outk==0) out_pack->set(c,toffset,nix,out_atoms.row_offset0(i),out_atoms.nrows0(i),out_atoms[i](common));
	  if(outk==1) out_pack->set(c,toffset,nix,out_atoms.row_offset1(i),out_atoms.nrows1(i),out_atoms[i](common));
	  if(outk==2) out_pack->set(c,toffset,nix,out_atoms.row_offset2(i),out_atoms.nrows2(i),out_atoms[i](common));

	  if(ink==0) in_pack->set(c,toffset,nix,in_atoms.row_offset0(j),in_atoms.nrows0(j),in_atoms[j](common));
	  if(ink==1) in_pack->set(c,toffset,nix,in_atoms.row_offset1(j),in_atoms.nrows1(j),in_atoms[j](common));
	  if(ink==2) in_pack->set(c,toffset,nix,in_atoms.row_offset2(j),in_atoms.nrows2(j),in_atoms[j](common));

	  if(gatherk==0) toffset+=1;
	  if(gatherk==1) toffset+nix;
	  if(gatherk==2) toffset+nix*nix;
	  c++;
	});

      R->outB=to_share(out_pack);
      R->inB=to_share(in_pack);
      */

