// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Cgraph
#define _Cgraph

#include "Ptens_base.hpp"
#include <map>

#include "Gdims.hpp"
#include "IntTensor.hpp"


namespace ptens{


  typedef vector<int> CgraphList; 


  class Cgraph{
  public:

    typedef cnine::IntTensor IntTensor;

    int maxi=0;
    int maxj=0;
    map<int,CgraphList*> lists;
    //vector<int> rstrides;
    //vector<int> xstrides;

    //mutable int n=0;
    mutable int* arrg=nullptr;
    mutable bool current=false;


    ~Cgraph(){
      for(auto p:lists) delete p.second;
      if(arrg) CUDA_SAFE(cudaFree(arrg));
    }


  public:

    Cgraph(){}
    

  public:

    Cgraph(const Cgraph& x){
      for(auto p:x.lists) 
	lists[p.first]=new CgraphList(*p.second);
    }


  public:


    static Cgraph random(const int n, const float p=0.5){
      Cgraph G;
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<n; i++) 
	for(int j=0; j<i; j++)
	  if(distr(rndGen)<p){
	    G.push(i,j);
	    G.push(j,i);
	  }
      return G;
    }

    static Cgraph from_list(const IntTensor& M){
      Cgraph R; 
      assert(M.ndims()==2);
      assert(M.dim(1)==2);
      int N=M.dim(0);
      for(int i=0; i<N; i++)
	R.push(M(i,0),M(i,1));
      return R;
    }

    static Cgraph from_matrix(const IntTensor& A){
      Cgraph R; 
      assert(A.ndims()==2);
      int N=A.dim(0);
      assert(A.dim(1)==N);
      for(int i=0; i<N; i++)
	for(int j=0; j<N; j++)
	  if(A(i,j)>0) R.push(i,j);
      return R;
    }


  public: // ---- access -----------------------------------------------------------------------------------

 
    void push(const int i, const int j){
      current=false;
      maxi=std::max(i+1,maxi);
      maxj=std::max(j+1,maxj);
      auto it=lists.find(i);
      if(it!=lists.end())
	it->second->push_back(j);
      else{
	CgraphList* lst=new CgraphList;
	lst->push_back(j);
	lists[i]=lst;
      }
    }


    void forall_edges(std::function<void(const int, const int)> lambda) const{
      for(auto& p: lists){
	int i=p.first;
	for(auto q: *p.second)
	  lambda(i,q);
      }
    }


    IntTensor tensor() const{
      int n=maxi;
      IntTensor R=IntTensor::zero({n,n});
      for(auto p:lists){
	int i=p.first;
	for(auto q: *p.second)
	  R.set(i,q,1);
      }
      return R;
    }


  public: // ---- GPU side ---------------------------------------------------------------------------------


    void prepare(const int dev) const{
      //#ifdef _WITH_CUDA
      if(current) return;
      int n=lists.size();
      if(arrg) CUDA_SAFE(cudaFree(arrg));
      int N=n;
      for(auto p:lists)
	N+=2+2*p.second->size();
      int* arr=new int[N];

      int i=0;
      int lp=n;
      for(auto p:lists){
	arr[i]=lp;
	auto lst=*p.second;
	arr[lp]=p.first;
	arr[lp+1]=lst.size();
	for(int j=0; j<lst.size(); j++){
	  arr[lp+2+j]=lst[j];
	}
	lp+=2+lst.size();
	i++;
      }

      CUDA_SAFE(cudaMalloc((void **)&arrg, N*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg,arr,N*sizeof(int),cudaMemcpyHostToDevice));
      delete[] arr;
      //#endif
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto it: lists){
	oss<<indent<<it.first<<"<-(";
	//for(auto p:it.second->lst)
	const CgraphList& lst=*it.second;
	for(int i=0; i<lst.size(); i++){
	  oss<<lst[i];
	  if(i<lst.size()-1) oss<<",";
	}
	oss<<")"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Cgraph& x){
      stream<<x.str(); return stream;}

  };


}

#endif 
