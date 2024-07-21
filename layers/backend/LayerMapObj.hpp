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

#ifndef _ptens_LayerMapObj
#define _ptens_LayerMapObj

#include "observable.hpp"
#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "flog.hpp"


namespace ptens{


  class LayerMapObj: public cnine::observable<LayerMapObj>, public cnine::map_of_lists<int,int>{
  public:

    typename cnine::map_of_lists<int,int> BASE;


    LayerMapObj():
      observable(this){}

    static shared_ptr<LayerMapObj> overlaps_map(const AtomsPackObj& out, const AtomsPackObj& in){
      cnine::flog timer("LayerMapObj::make_overlaps");
      auto R=new LayerMapObj();

      if(in.size()<10){
	for(int i=0; i<out.size(); i++){
	  auto v=(out)(i);
	  for(int j=0; j<in.size(); j++){
	    auto w=in(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      R->push_back(i,j);
	  }
	}
      }

      else{
	unordered_map<int,vector<int> > map;
	for(int j=0; j<in.size(); j++){
	  auto w=(in)(j);             
	  for(auto p:w){
	    auto it=map.find(p);
	    if(it==map.end()) map[p]=vector<int>({j});
	    else it->second.push_back(j);
	  }
	}          
	for(int i=0; i<out.size(); i++){
	  auto v=out(i);
	  for(auto p:v){
	    auto it=map.find(p);
	    if(it!=map.end())
	      for(auto q:it->second)
		R->push_back(i,q);
	  }
	}
      }

      auto r=cnine::to_share(R);
      //out.related_layermaps.emplace_back(r);
      return r;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "LayerMapObj";
    }

    string repr() const{
      return "<LayerMapObj>";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:*this){
	oss<<indent<<p.first<<"<-(";
	for(int i=0; i<p.second.size()-1; i++)
	  oss<<p.second[i]<<",";
	if(p.second.size()>0) 
	  oss<<p.second.back();
	oss<<")"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LayerMapObj& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
