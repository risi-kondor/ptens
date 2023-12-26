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

#ifndef _ptens_MessageListObj
#define _ptens_MessageListObj
#define _ptens_TransferMapObj

#include "Tensor.hpp"
#include "array_pool.hpp"
#include "AindexPack.hpp"
#include "GatherMap.hpp"
#include "TransferMapGradedObj.hpp"
#include "flog.hpp"
#include "map_of_lists.hpp"
#include "map_of_maps.hpp"
#include "hlists.hpp"


namespace ptens{

  template<typename DUMMY> class AtomsPack0obj;
  template<typename DUMMY> class AtomsPack1obj;
  template<typename DUMMY> class AtomsPack2obj;


  class MessageListObj: public cnine::observable<MessageListObj>{
  public:


    cnine::hlists<int> in;
    cnine::hlists<int> out;

    mutable shared_ptr<AtomsPack0obj<int> > source0;
    mutable shared_ptr<AtomsPack1obj<int> > source1;
    mutable shared_ptr<AtomsPack2obj<int> > source2;

    MessageListObj():
      observable(this){}


  public: // ---- Access -------------------------------------------------------------------------------------


    pair<const cnine::hlists<int>&, const cnine::hlists<int>&> lists() const{
      return pair<const cnine::hlists<int>&, const cnine::hlists<int>&>(in,out);
    }
    

  public: // ---- Constructors -------------------------------------------------------------------------------


    // overlaps 
    MessageListObj(const cnine::array_pool<int>& y, const cnine::array_pool<int>& x):
      observable(this){
      cout<<"overlaps"<<endl;
      cnine::flog timer("MessageListObj::[overlaps]");

      if(x.size()<10){
	for(int i=0; i<y.size(); i++){
	  auto v=y(i);
	  for(int j=0; j<x.size(); j++){
	    auto w=x(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      append_intersection(i,j,y.view_of(i),x.view_of(j));
	  }
	}
	return;
      }

      cnine::map_of_lists<int,int> in_lists;
      int nx=x.size();
      for(int i=0; i<nx; i++)
	x.for_each_of(i,[&](const int j){in_lists.push_back(j,i);});

      cnine::map_of_maps<int,int,bool> done;
      int ny=y.size();
      for(int i=0; i<ny; i++)
	y.for_each_of(i,[&](const int j){
	    for(auto p:in_lists[j])
	      if(!done.is_filled(p,i)){
		append_intersection(p,i,x.view_of(p),y.view_of(i));
		done.set(p,i,true);
	      }
	  });
    }


    void append_intersection(const int xi, const int yi, const cnine::Itensor1_view& x, const cnine::Itensor1_view& y){
      vector<int> v_in;
      vector<int> v_out;
      for(int i=0; i<x.n0; i++){
	int t=x(i);
	for(int j=0; j<y.n0; j++)
	  if(y(j)==t){
	    v_in.push_back(i); 
	    v_out.push_back(j); 
	    break;
	  }
      }
      in.push_back(xi,v_in);
      out.push_back(yi,v_out);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "MessageListObj";
    }

    string repr() const{
      return "MessageListObj";
    }

    string str(const string indent="") const{
      ostringstream oss;
      PTENS_ASSRT(in.size()==out.size());
      for(int m=0; m<in.size(); m++){
	oss<<indent<<out.head(m)<<":(";
	for(int i=0; i<out.size_of(m); i++)
	  oss<<out(m,i)<<",";
	if(out.size_of(m)>0) oss<<"\b";
	oss<<") <- ";
	oss<<indent<<in.head(m)<<":(";
	for(int i=0; i<in.size_of(m); i++)
	  oss<<in(m,i)<<",";
	if(in.size_of(m)>0) oss<<"\b";
	oss<<")"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const MessageListObj& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
