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
 */

#ifndef _Ptens_GraphCache
#define _Ptens_GraphCache

#include "GgraphObj.hpp"


namespace ptens{

  class GgraphCache: public std::unordered_map<int,shared_ptr<GgraphObj> >{
  public:

    typedef std::unordered_map<int,shared_ptr<GgraphObj> > BASE;
    
    unordered_map<cnine::Tensor<int>,int> edge_list_map;


    int next_id=0;


  public: // ---- Access -------------------------------------------------------------------------------------


    int size(){
      return BASE::size();
    }

    void cache(const int key, const shared_ptr<GgraphObj>& x){
      (*this)[key]=x;
      //x->is_cached=true;
      edge_list_map[x->edge_list()]=key;
    }

    shared_ptr<GgraphObj> operator()(const int key){
      auto it=find(key);
      if(it==BASE::end())
	throw std::out_of_range("ptens error in "+string(__PRETTY_FUNCTION__)+": index "+to_string(key)+" not found in Ggraph cache.");
      return it->second;
      //return Ggraph(it->second);
    }

    shared_ptr<GgraphObj> from_edge_list(const int key, const cnine::Tensor<int>& edges){
      auto it=find(key);
      if(it!=BASE::end()) return it->second;
      shared_ptr<GgraphObj> r(new GgraphObj(GgraphObj::from_edges(edges)));
      (*this)[key]=r;
      edge_list_map[edges]=key;
      next_id=std::max(next_id,key+1);
      return r;
      //return Ggraph(r);
    }

    std::pair<int,shared_ptr<GgraphObj> > from_edge_list(const cnine::Tensor<int>& edges){
      auto it=edge_list_map.find(edges);
      if(it!=edge_list_map.end()){
	cout<<"Found"<<endl;
	auto it2=find(it->second);
	PTENS_ASSRT(it2!=BASE::end());
	return make_pair(it->second,it2->second);
      }
      shared_ptr<GgraphObj> r(new GgraphObj(GgraphObj::from_edges(edges)));
      cout<<"Add to cache"<<endl;
      (*this)[next_id]=r;
      edge_list_map[edges]=next_id;
      next_id++;
      return make_pair(next_id-1,r);
      //return make_pair(next_id-1,Ggraph(r));
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------

    
    string str(const string indent=""){
      ostringstream oss;
      for(auto& p: *this){
	oss<<indent<<"Graph "<<p.first<<":"<<endl;
	oss<<p.second->str(indent+"  ");
      }
      return oss.str();
    }

  };

}

#endif 
