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

#ifndef _ptens_SubgraphObj
#define _ptens_SubgraphObj

#include "Ptens_base.hpp"
#include "SparseRmatrix.hpp"
#include "Hgraph.hpp"
#include "Tensor.hpp"
#include "SymmEigendecomposition.hpp"


namespace ptens{


  class SubgraphObj: public Hgraph /*public cnine::SparseRmatrix*/{
  public:

    typedef Hgraph BASE;
    typedef cnine::SparseRmatrix BaseMatrix;
    typedef cnine::Tensor<float> rtensor;

    //using BaseMatrix::BaseMatrix;
    using BASE::BASE;

    cnine::Tensor<float> evecs;
    vector<int> eblocks;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SubgraphObj(const int _n):
      SubgraphObj(_n,_n){}

    SubgraphObj(const vector<pair<int,int> >& list): 
      SubgraphObj([](const vector<pair<int,int> >& list){
	  int t=0; for(auto& p: list) t=std::max(std::max(p.first,p.second),t); return t+1;}(list)){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    SubgraphObj(const int _n, const initializer_list<pair<int,int> >& list): 
      SubgraphObj(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    SubgraphObj(const int _n, const initializer_list<pair<int,int> >& list, const RtensorA& _labels): 
      Hgraph(_n,list,_labels){}

    //SubgraphObj(const cnine::RtensorA& A):
    //SubgraphObj(A){}


    SubgraphObj(const int n, const cnine::RtensorA& _edges):
      SubgraphObj(_edges,n){}
    //PTENS_ASSRT(_edges.ndims()==2);
    //PTENS_ASSRT(_edges.get_dim(0)==2);
    //PTENS_ASSRT(_edges.max()<n);
    //int nedges=_edges.get_dim(1);
    //for(int i=0; i<nedges; i++)
    //set(_edges(0,i),_edges(1,i),1.0);
    //}

    SubgraphObj(const int n, const cnine::RtensorA& _edges, const cnine::RtensorA& _labels):
      Hgraph(_edges,_labels,n){}

    SubgraphObj(const int n, const cnine::RtensorA& _edges, const cnine::RtensorA& _evecs, const cnine::RtensorA& evals):
      Hgraph(_edges,n), evecs(_evecs){
      make_eblocks(evals);}

    SubgraphObj(const int n, const cnine::RtensorA& _edges, const cnine::RtensorA& _labels, const cnine::RtensorA& _evecs, const cnine::RtensorA& evals):
      Hgraph(_edges,_labels,n), evecs(_evecs){
      make_eblocks(evals);}


  public: 


    bool has_espaces() const{
      return eblocks.size()>0;
    }

    void make_eigenbasis(){
      if(eblocks.size()>0) return;
      int n=getn();

      cnine::Tensor<float> L=cnine::Tensor<float>::zero({n,n});
      L.view2().add(dense().view2()); 
      for(int i=0; i<n; i++){
	float t=0; 
	for(int j=0; j<n; j++) t+=L(i,j);
	L.inc(i,i,-t);
      }

      auto eigen=cnine::SymmEigendecomposition<float>(L);
      evecs=eigen.U();
      make_eblocks(eigen.lambda());
    }

    void set_evecs(const cnine::Tensor<float>& _evecs, const cnine::Tensor<float>& _evals) const{
      const_cast<SubgraphObj&>(*this).evecs=_evecs;
      const_cast<SubgraphObj&>(*this).make_eblocks(_evals);
    }

    void make_eblocks(const cnine::Tensor<float>& evals){
      PTENS_ASSRT(evals.dims.size()==1);
      PTENS_ASSRT(getn()==evals.dims[0]);
      eblocks.clear();
      int j=0;
      for(int i=0; i<evals.dims[0];){
	float t=evals(i);
	int start=i;
	while(i<evals.dims[0] && std::abs(evals(i)-t)<10e-5) i++;
	eblocks.push_back(i-start);
      }
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "SubgraphObj";
    }

  };

}


namespace std{
  template<>
  struct hash<ptens::SubgraphObj>{
  public:
    size_t operator()(const ptens::SubgraphObj& x) const{
      if(x.is_labeled) return (hash<cnine::SparseRmatrix>()(x)<<1)^hash<cnine::RtensorA>()(x.labels);
      return hash<cnine::SparseRmatrix>()(x);
    }
  };
}


#endif 
