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

#include "sparse_graph.hpp"
//#include "PtensFindPlantedSubgraphs.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "SubgraphObj.hpp"
#include "RtensorA.hpp"

namespace ptens{


  class GgraphObj: public cnine::sparse_graph<int,float,float>{
  public:

    typedef cnine::sparse_graph<int,float,float> BASE;

    mutable unordered_map<SubgraphObj,cnine::array_pool<int>*> subgraphlist_cache;
    mutable unordered_map<SubgraphObj,shared_ptr<cnine::Tensor<int> > > subgraphlistmx_cache;

    using BASE::BASE;


    GgraphObj(const initializer_list<pair<int,int> >& list, const int n): 
      BASE(n,list){}

    //GgraphObj(const cnine::RtensorA& M):
    //obj(new Hgraph(M)){}


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
      BASE(x){}


  public: // ---- Access --------------------------------------------------------------------------------------


    //int getn() const{
    //return obj->getn();
    //}

    //cnine::RtensorA dense() const{
    //return obj->dense();
    //}

    //bool operator==(const GgraphObj& x) const{
    //return obj==x.obj;
    //}


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
