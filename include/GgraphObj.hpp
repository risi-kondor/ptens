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

#ifndef _Ptens_GraphObj
#define _Ptens_GraphObj

#include "once.hpp"
#include "sparse_graph.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "FindPlantedSubgraphs2.hpp"
#include "SubgraphObj.hpp"
#include "RtensorA.hpp"
#include "AtomsPack.hpp"


namespace ptens{


  class GgraphObj: public cnine::sparse_graph<int,float,float>{
  public:

    typedef cnine::sparse_graph<int,float,float> BASE;

    mutable unordered_map<SubgraphObj,AtomsPack> subgraphpack_cache;

    using BASE::BASE;


    GgraphObj(const initializer_list<pair<int,int> >& list, const int n): 
      BASE(n,list){}


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static GgraphObj random(const int _n, const float p=0.5){
      return BASE::random(_n,p);
    }


    // eliminate this eventually
    GgraphObj(const int n, const cnine::RtensorA& M):
      BASE(n){
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.dim(0)==2);
      for(int i=0; i<M.dims(1); i++)
	set(M(0,i),M(1,i),1.0);
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    GgraphObj(const BASE& x):
      BASE(x){
      //cout<<"conversion"<<endl;
    }

    //GgraphObj(const GgraphObj& x):
    //BASE(x){
    //cout<<"GGraph copied"<<endl;
    //}


  public: // ---- Access --------------------------------------------------------------------------------------


  public: // ---- Operations ---------------------------------------------------------------------------------


    GgraphObj permute(const cnine::permutation pi) const{
      cnine::Tensor<int> A({2,nedges()},cnine::fill_zero());
      int t=0;
      for_each_edge([&](const int i, const int j, const float v){
	  A.set(0,t,(float)pi(i));
	  A.set(1,t,(float)pi(j));
	  t++;
	});
      return BASE(getn(),A);
     }


    AtomsPack subgraphs(const SubgraphObj& H){
      auto it=subgraphpack_cache.find(H);
      if(it!=subgraphpack_cache.end()) return it->second;
      else{
	//cout<<"Not in subgraph cache"<<endl;
	cnine::flog timer("GgraphObj::finding subgraphs");
	subgraphpack_cache[H]=AtomsPack(new AtomsPackObj
	  (cnine::Tensor<int>(cnine::FindPlantedSubgraphs<float>(*this,H))));
	return subgraphpack_cache[H];
      }
    }

    cnine::once<AtomsPack> edges=cnine::once<AtomsPack>([&](){
	auto R=new AtomsPackObj(nedges(),2,cnine::fill_raw());
	int i=0;
	for(auto& p:data)
	  for(auto& q:p.second)
	    if(p.first<q.first){
	      R->set(i,0,p.first);
	      R->set(i,1,q.first);
	      i++;
	    }
	return AtomsPack(R);
      });
    

  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "ptens::GgraphObj";
    }

    //string str(const string indent="") const{
    //return obj->str(indent);
    //}

    friend ostream& operator<<(ostream& stream, const GgraphObj& x){
      stream<<x.str(); return stream;}


  };

}

#endif 


    // deprecated 
    /*
    cnine::array_pool<int> subgraphs_list(const SubgraphObj& H){
      auto it=subgraphlist_cache.find(H);
      if(it!=subgraphlist_cache.end()) return *it->second;
      auto newpack=new cnine::array_pool<int>(cnine::FindPlantedSubgraphs(*this,H));
      subgraphlist_cache[H]=newpack;
      return *newpack;
    }
    */
    /*
    cnine::Tensor<int>& subgraphs_matrix(const SubgraphObj& H){
      cnine::flog timer("CachedPlantedSubgraphsMx");
      auto it=subgraphlistmx_cache.find(H);
      if(it!=subgraphlistmx_cache.end()) return *it->second;
      else{
	shared_ptr<cnine::Tensor<int> > p(new cnine::Tensor<int>(cnine::FindPlantedSubgraphs<float>(*this,H)));
	subgraphlistmx_cache[H]=p;
	return *p;
      }
    }
    */
