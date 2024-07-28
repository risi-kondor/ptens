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

#ifndef _ptens_PtensorMapGradedObj
#define _ptens_PtensorMapGradedObj

#include "SparseRmatrix.hpp"
#include "Tensor.hpp"
#include "array_pool.hpp"
#include "AindexPack.hpp"
//#include "GatherMap.hpp"
#include "flog.hpp"


namespace ptens{


  //template<typename ATOMSPACK> // dummy template to avoid circular dependency 
  class PtensorMapGradedObj: public cnine::SparseRmatrix{
  public:
    
    typedef cnine::SparseRmatrix SparseRmatrix;
    typedef cnine::Tensor<int> ITENSOR;

    using SparseRmatrix::SparseRmatrix;

    int k;
    ITENSOR in;
    ITENSOR out;

    //mutable shared_ptr<cnine::GatherMap> bmap;

    ~PtensorMapGradedObj(){
    }

    PtensorMapGradedObj(const int _k, const int _n, const int _m):
      k(_k),
      SparseRmatrix(_n,_m){}


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      for(auto q:lists){
	if(q.second->size()>0)
	  return false;
      }
      return true;
    }

    void for_each_edge(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }


  public: // ---- Intersects --------------------------------------------------------------------------------------------


    void make_intersects(const AtomsPackObj& in_pack, const AtomsPackObj& out_pack){
      cnine::ftimer timer("PtensorMapGradedObj["+to_string(k)+"]::make_intersects");

      in=ITENSOR(cnine::Gdims(size(),k+1),cnine::fill_raw());
      out=ITENSOR(cnine::Gdims(size(),k+1),cnine::fill_raw());

      int p=0;
      for_each_edge([&](const int i, const int j, const float v){
	  cnine::Itensor1_view in_atoms=in_pack.view_of(j);
	  cnine::Itensor1_view out_atoms=out_pack.view_of(i);
	  in.set(p,0,j);
	  out.set(p,0,i);
	  int n_in=in_atoms.n0;
	  int n_out=out_atoms.n0;
	  int nfound=0;
	  for(int a=0; a<n_in && nfound<k; a++){
	    int ix=in_atoms(a);
	    for(int b=0; b<n_out && nfound<k; b++){
	      if(out_atoms(b)==ix){
		in.set(p,nfound+1,a);
		out.set(p,nfound+1,b);
		nfound++;
		break;
	      }
	    }
	  }
	  p++;
	}, false);
      //get_bmap();
    }

    /*
    std::shared_ptr<cnine::GatherMap> get_bmap() const{
      if(bmap) return bmap; 

      int nlists=0;
      int nedges=0;
      for(auto q:lists)
	if(q.second->size()>0){
	  nlists++;
	  nedges+=q.second->size();
	}
      
      cnine::GatherMap* R=new cnine::GatherMap(nlists,nedges);
      int i=0;
      int m=0;
      int tail=3*nlists;
      for(auto q:lists){
	int len=q.second->size();
	if(len==0) continue;
	R->arr[3*i]=tail;
	R->arr[3*i+1]=len;
	R->arr[3*i+2]=q.first;
	int j=0;
	for(auto p:*q.second){
	  R->arr[tail+2*j]=m++;
	  *reinterpret_cast<float*>(R->arr+tail+2*j+1)=p.second;
	  j++;
	}
	tail+=2*len;
	i++;
      }
      
      bmap=std::shared_ptr<cnine::GatherMap>(R);
      return bmap;
    }
    */
    
  };

}

#endif 


