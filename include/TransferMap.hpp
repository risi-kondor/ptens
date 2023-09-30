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

#ifndef _ptens_TransferMap
#define _ptens_TransferMap

#include "SparseRmatrix.hpp"
#include "Tensor.hpp"
#include "array_pool.hpp"
#include "AindexPack.hpp"
#include "GatherMap.hpp"
#include "ftimer.hpp"


namespace ptens{

  class TransferMap: public cnine::SparseRmatrix{
  public:
    
    typedef cnine::SparseRmatrix SparseRmatrix;
    typedef cnine::Tensor<int> IntMatrix;

    using SparseRmatrix::SparseRmatrix;

    mutable shared_ptr<cnine::GatherMap> bmap;


  public: // ---- Construct from overlaps ------------------------------------------------------------------------------


    TransferMap(const cnine::Tensor<int>& y, const cnine::Tensor<int>& x):
      TransferMap(x.dim(1),y.dim(1)){
      cnine::ftimer timer("TransferMap::TransferMap(const Tensor<int>&, const Tensor<int>&)");
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(y.ndims()==2);
      const int kx=x.dims[1];
      const int ky=y.dims[1];

      for(int i=0; i<x.dims[0]; i++){
	for(int j=0; j<y.dims[0]; j++){

	  bool found=false;
	  for(int a=0; !found && a<kx; a++){
	    int t=x(i,a);
	    for(int b=0; !found && b<ky; b++)
	      if(y(j,b)==t) found=true;
	  }
	  if(found) set(i,j,1);

	}
      }
    }


    TransferMap(const cnine::Tensor<int>& y, const cnine::array_pool<int>& x):
      TransferMap(x.size(),y.dims[0]){
      cnine::ftimer timer("TransferMap::TransferMap(const Tensor<int>&, const AtomsPack&)");
      CNINE_ASSRT(y.ndims()==2);
      const int ky=y.dims[1];

      for(int i=0; i<x.size(); i++){
	auto v=x(i);
	for(int j=0; j<y.dims[0]; j++){
	  
	  bool found=false;
	  for(int a=0; !found && a<v.size(); a++){
	    int t=v[a];
	    for(int b=0; !found && b<ky; b++)
	      if(y(j,b)==t) found=true;
	  }
	  if(found) set(i,j,1);
	  
	}
      }
    }

      
    TransferMap(const cnine::array_pool<int>& y, const cnine::Tensor<int>& x):
      TransferMap(x.dims[0],y.size()){
      cnine::ftimer timer("TransferMap::TransferMap(const AtomsPack&, const Tensor<int>&)");
      CNINE_ASSRT(x.ndims()==2);
      const int kx=x.dims[1];

	for(int i=0; i<x.dims[0]; i++){
	  for(int j=0; j<y.size(); j++){
	    auto v=y(j);
	    
	    bool found=false;
	    for(int a=0; !found && a<kx; a++){
	      int t=x(i,a);
	      for(int b=0; !found && b<v.size(); b++)
		if(v[b]==t) found=true;
	    }
	    if(found) set(i,j,1);
	  }
	}
    }


    TransferMap(const cnine::array_pool<int>& y, const cnine::array_pool<int>& x):
      TransferMap(x.size(),y.size()){
      cnine::ftimer timer("TransferMap::TransferMap(const AtomsPack&, const AtomsPack&)");
      if(y.size()<10){
	for(int i=0; i<x.size(); i++){
	  auto v=x(i);
	  for(int j=0; j<y.size(); j++){
	    auto w=y(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      set(i,j,1.0);
	  }
	}
      }else{
	unordered_map<int,vector<int> > map;
	for(int j=0; j<y.size(); j++){
	  auto w=y(j);             
	  for(auto p:w){
	    auto it=map.find(p);
	    if(it==map.end()) map[p]=vector<int>({j});
	    else it->second.push_back(j);
	  }
	}          
	for(int i=0; i<x.size(); i++){
	  auto v=x(i);
	  for(auto p:v){
	    auto it=map.find(p);
	    if(it!=map.end())
	      for(auto q:it->second)
		set(i,q,1.0);
	  }
	}
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //int getn() const{
    //return n;
    //}

    bool is_empty() const{
      for(auto q:lists)
	if(q.second->size()>0)
	  return false;
      return true;
    }

    void forall_edges(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }


  public: // ---- Intersects --------------------------------------------------------------------------------------------

    // for future use 
    /*  
    pair<IntMatrix,IntMatrix> intersects(const IntMatrix& inputs, const IntMatrix& outputs, const bool self=0) const{
      PTENS_ASSRT(outputs.dims[0]==n);
      PTENS_ASSRT(inputs.dims[0]==m);
      int N=size();
      int kin=inputs.dims[1];
      int kout=outputs.dims[1];

      IntMatrix in_indices({N,kin,kin},cnine::fill_zero());
      IntMatrix out_indices({N,kout,kout},cnine::fill_zero());
      int t=0;
      forall_edges([&](const int i, const int j, const float v){
	  for(int a=0; a<kin; a++){
	    int x=inputs(j,a);
	    int s=0;
	    for(int b=0; b<kout; b++){
	      if(outputs(i,b)==x){
		in_indices.set(t,s,a,1.0);
		out_indices.set(t,s,b,1.0);
		s++;
		break;
	      }
	    }
	  }
	  t++;
	});
      out_indices.bmap=get_bmap();
      return make_pair(in_indices, out_indices);
    }
    */

    pair<AindexPack,AindexPack> intersects(const AtomsPack& inputs, const AtomsPack& outputs, const bool self=0) const{
      cnine::ftimer timer("TransferMap::intersects");
      //cout<<n<<" "<<m<<" "<<inputs.size()<<" "<<outputs.size()<<endl;
      PTENS_ASSRT(outputs.size()==n);
      PTENS_ASSRT(inputs.size()==m);
      AindexPack in_indices;
      AindexPack out_indices;
      forall_edges([&](const int i, const int j, const float v){
	  Atoms in=inputs[j];
	  Atoms out=outputs[i];
	  Atoms common=out.intersect(in);
	  //in_indices.push_back(j,in(common));
	  //out_indices.push_back(i,out(common));
	  auto _in=in(common);
	  auto _out=out(common);
	  in_indices.push_back(j,_in);
	  out_indices.push_back(i,_out);
	  in_indices.count1+=_in.size();
	  in_indices.count2+=_in.size()*_in.size();
	  out_indices.count1+=_out.size();
	  out_indices.count2+=_out.size()*_out.size();
	    
	}, self);
      out_indices.bmap=get_bmap();
      return make_pair(in_indices, out_indices);
    }


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

    
  };

}

#endif 
