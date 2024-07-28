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

#ifndef _ptens_PtensorMapObj
#define _ptens_PtensorMapObj

#include "observable.hpp"
//#include "SparseRmatrix.hpp"
#include "Tensor.hpp"
#include "array_pool.hpp"
#include "map_of_lists.hpp"
#include "AindexPack.hpp"
#include "AindexPackB.hpp"
#include "GatherMapB.hpp"
#include "PtensorMapGradedObj.hpp"
#include "flog.hpp"


namespace ptens{


  class PtensorMapObj: //public cnine::SparseRmatrix, 
		       public cnine::observable<PtensorMapObj>{
  public:
    
    //typedef cnine::SparseRmatrix SparseRmatrix;
    //typedef cnine::SparseRmatrix BASE;
    //typedef cnine::Tensor<int> IntMatrix;

    //using SparseRmatrix::SparseRmatrix;

    cnine::map_of_lists<int,int> gmap;

    shared_ptr<AtomsPackObj> atoms;
    shared_ptr<AindexPack> in;
    shared_ptr<AindexPack> out;
    //shared_ptr<AindexPackB> inB;
    //shared_ptr<AindexPackB> outB;
    int n_in, n_out;

    //mutable shared_ptr<cnine::GatherMap> bmap;
    mutable shared_ptr<cnine::GatherMapB> bmap;

    unordered_map<int,unique_ptr<PtensorMapGradedObj> > graded_maps;


    ~PtensorMapObj(){
      //cout<<"Destroying a PtensorMapObj"<<endl;
    }

    PtensorMapObj():
      observable(this),
      atoms(new AtomsPackObj()),
      in(new AindexPack()),
      out(new AindexPack()){}

    /*
    PtensorMapObj(const AtomsPackObj& _in_atoms, const AtomsPackObj& _out_atoms, const bool graded=false):
      //SparseRmatrix(_out_atoms.size(),_in_atoms.size()),
      observable(this),
      atoms(new AtomsPackObj()),
      in(new AindexPack()),
      out(new AindexPack()),
      n_in(_in_atoms.size()),
      n_out(_out_atoms.size()){
      //cout<<"Creating new PtensorMapObj...";//<<endl;
      if(graded){
	make_graded(_in_atoms,_out_atoms);
      }else{
	make_overlaps(_in_atoms,_out_atoms);
	make_intersects(_in_atoms,_out_atoms);
      }
      //cout<<"done."<<endl;
    }
    */

  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      return false; 
      //for(auto q:lists){
      //if(q.second->size()>0)
      //  return false;
      //}
      //return true;
    }

    bool is_graded() const{
      return graded_maps.size()>0;
    }

    int ntotal() const{
      return gmap.size();
      //return BASE::size();
    }

    pair<const AindexPack&, const AindexPack&> ipacks() const{
      return pair<const AindexPack&, const AindexPack&>(*in,*out);
    }

    //pair<const cnine::hlists<int>&, const cnine::hlists<int>&> ipacks() const{
    //return pair<const cnine::hlists<int>&, const cnine::hlists<int>&>(in,out);
    //}

//     void for_each_row(std::function<void(const int, const vector<int>)> lambda) const{
//       for(auto& p: lists){
// 	vector<int> v;
// 	p.second->forall_nonzero([&](const int j, const float a){
// 	    v.push_back(j);});
// 	lambda(p.first,v);
//       }
//     }

//     void for_each_edge(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
//       for(auto& p: lists){
// 	int i=p.first;
// 	if(self) lambda(i,i,1.0);
// 	p.second->forall_nonzero([&](const int j, const float v){
// 	    lambda(i,j,v);});
//       }
//     }

//    size_t rmemsize() const{
//      return BASE::rmemsize();
//    }


  public: // ---- Overlaps -----------------------------------------------------------------------------------


    void make_overlaps(const AtomsPackObj& in_atoms, const AtomsPackObj& out_atoms){
      cnine::flog timer("PtensorMapObj::make_overlaps");
      if(in_atoms.size()<10){
	for(int i=0; i<out_atoms.size(); i++){
	  auto v=(out_atoms)(i);
	  for(int j=0; j<in_atoms.size(); j++){
	    auto w=(in_atoms)(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      gmap.push_back(i,j);
	    //set(i,j,1.0);
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
	    //set(i,q,1.0);
	  }
	}
      }
    }




  public: // ---- Intersects --------------------------------------------------------------------------------------------


    void make_intersects(const AtomsPackObj& in_atoms, const AtomsPackObj& out_atoms){
      cnine::ftimer timer("PtensorMapObj::make_intersects");

      //PTENS_ASSRT(out_atoms.size()==n);
      //PTENS_ASSRT(in_atoms.size()==m);
      //for_each_edge([&](const int i, const int j, const float v){
      gmap.for_each([&](const int i, const int j){
	  Atoms in_j=(in_atoms)[j];
	  Atoms out_i=(out_atoms)[i];
	  Atoms common=out_i.intersect(in_j);
	  auto _in=in_j(common);
	  auto _out=out_i(common);
	  atoms->push_back(common);
	  in->push_back(j,_in);
	  out->push_back(i,_out);
	  in->count1+=_in.size();
	  in->count2+=_in.size()*_in.size();
	  out->count1+=_out.size();
	  out->count2+=_out.size()*_out.size();
	    
	});
      out->bmap2=get_bmap();
    }


    std::shared_ptr<cnine::GatherMapB> get_bmap() const{
      if(bmap.get()) return bmap; 
      cnine::GatherMapB* R=new cnine::GatherMapB(n_out,n_in);
      int m=0;
      // TODO 
      //for(auto q:lists){
      //vector<int> v;
      //for(auto p:*q.second)
      //  v.push_back(m++);
      //R->arr.push_back(q.first,v);
      //}
      bmap=to_share(R);
      return bmap;
    }


  public: // ---- Graded --------------------------------------------------------------------------------------------


    void make_graded(const AtomsPackObj& in_atoms, const AtomsPackObj& out_atoms){
      cnine::flog timer("PtensorMapObj::make_graded");

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
		it=graded_maps.emplace(k,unique_ptr<PtensorMapGradedObj>
		  (new PtensorMapGradedObj(k,out_atoms.size(),in_atoms.size()))).first;
	      it->second->set(i,j,1.0);
	    }
	  }
	}
      }

      for(auto& p:graded_maps){
	p.second->make_intersects(in_atoms,out_atoms);
      }
    }


    string str(const string indent=""){
      return gmap.str(indent);
    }
    
  };

}

#endif 


