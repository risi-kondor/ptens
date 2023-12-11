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

#ifndef _ptens_TransferMapObj
#define _ptens_TransferMapObj

#include "SparseRmatrix.hpp"
#include "Tensor.hpp"
#include "array_pool.hpp"
#include "AindexPack.hpp"
#include "GatherMap.hpp"
#include "TransferMapGradedObj.hpp"
#include "flog.hpp"


namespace ptens{


  template<typename ATOMSPACK> // dummy template to avoid circular dependency 
  class TransferMapObj: public cnine::SparseRmatrix{
  public:
    
    typedef cnine::SparseRmatrix SparseRmatrix;
    typedef cnine::Tensor<int> IntMatrix;

    using SparseRmatrix::SparseRmatrix;


    //shared_ptr<const ATOMSPACK> in_atoms;
    //shared_ptr<const ATOMSPACK> out_atoms;

    shared_ptr<AindexPack> in;
    shared_ptr<AindexPack> out;

    mutable shared_ptr<cnine::GatherMap> bmap;

    unordered_map<int,unique_ptr<TransferMapGradedObj<ATOMSPACK> > > graded_maps;


    ~TransferMapObj(){
      //cout<<"Destroying a TransferMapObj"<<endl;
    }

    TransferMapObj(const ATOMSPACK& _in_atoms, const ATOMSPACK& _out_atoms, const bool graded=false):
      SparseRmatrix(_out_atoms.size(),_in_atoms.size()),
      in(new AindexPack()),
      out(new AindexPack()){
      //cout<<"Creating new TransferMapObj...";//<<endl;
      if(graded){
	make_graded(_in_atoms,_out_atoms);
      }else{
	make_overlaps(_in_atoms,_out_atoms);
	make_intersects(_in_atoms,_out_atoms);
      }
      //cout<<"done."<<endl;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      for(auto q:lists){
	if(q.second->size()>0)
	  return false;
      }
      return true;
    }

    bool is_graded() const{
      return graded_maps.size()>0;
    }

    void for_each_edge(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }


  public: // ---- Overlaps -----------------------------------------------------------------------------------


    void make_overlaps(const ATOMSPACK& in_atoms, const ATOMSPACK& out_atoms){
      cnine::flog timer("TransferMapObj::make_overlaps");
      if(in_atoms.size()<10){
	for(int i=0; i<out_atoms.size(); i++){
	  auto v=(out_atoms)(i);
	  for(int j=0; j<in_atoms.size(); j++){
	    auto w=(in_atoms)(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      set(i,j,1.0);
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
		set(i,q,1.0);
	  }
	}
      }
    }




  public: // ---- Intersects --------------------------------------------------------------------------------------------


    void make_intersects(const ATOMSPACK& in_atoms, const ATOMSPACK& out_atoms){
      cnine::ftimer timer("TransferMapObj::make_intersects");

      PTENS_ASSRT(out_atoms.size()==n);
      PTENS_ASSRT(in_atoms.size()==m);
      for_each_edge([&](const int i, const int j, const float v){
	  Atoms in_j=(in_atoms)[j];
	  Atoms out_i=(out_atoms)[i];
	  Atoms common=out_i.intersect(in_j);
	  auto _in=in_j(common);
	  auto _out=out_i(common);
	  in->push_back(j,_in);
	  out->push_back(i,_out);
	  in->count1+=_in.size();
	  in->count2+=_in.size()*_in.size();
	  out->count1+=_out.size();
	  out->count2+=_out.size()*_out.size();
	    
	}, false);
      out->bmap=get_bmap();
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


  public: // ---- Graded --------------------------------------------------------------------------------------------


    void make_graded(const ATOMSPACK& in_atoms, const ATOMSPACK& out_atoms){
      cnine::flog timer("TransferMapObj::make_graded");

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
	  if(it!=map.end()){
	    for(auto j:it->second){
	      int k=out_atoms.n_intersects(in_atoms,i,j);
	      auto it=graded_maps.find(k);
	      if(it==graded_maps.end())
		it=graded_maps.emplace(k,unique_ptr<TransferMapGradedObj<ATOMSPACK> >
		  (new TransferMapGradedObj<ATOMSPACK>(k,out_atoms.size(),in_atoms.size()))).first;
	      it->second->set(i,j,1.0);
	    }
	  }
	}
      }

      for(auto& p:graded_maps){
	p.second->make_intersects(in_atoms,out_atoms);
      }
    }

    
  };

}

#endif 


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
    //pair<AindexPack,AindexPack> intersects(const ATOMSPACK& inputs, const ATOMSPACK& outputs, const bool self=0) const{
