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

#ifndef _ptens_PtensorTransferListObj
#define _ptens_PtensorTransferListObj
#define _ptens_TransferMapObj

#include "Tensor.hpp"
#include "array_pool.hpp"
#include "AindexPack.hpp"
#include "GatherMap.hpp"
#include "TransferMapGradedObj.hpp"
#include "flog.hpp"


namespace ptens{

  class PtensorTransferListObj{
  public:

    //typedef cnine::hlists<pair<int,int> > BASE;

    hlist<int> in;
    hlist<int> out;


  public: // ---- Access -------------------------------------------------------------------------------------


    pair<const hlist<int>&, const hlist<int>&> lists const{
      return pair<const hlist<int>&, const hlist<int>&>(in,out);
    }

    void append_intersection(const int xi, const int yi, const cnine::Itensor1_view& x, const cnine::Itensor1_view& y){
      vector<int> v_in;
      vector<int> v_out;
      for(int i=0; i<x.n0; i++){
	int t=x(i);
	for(int j=0; j<y.n0; j++)
	  if(y(j)==t){
	    v_in.push_back(i); 
	    v_out.push_back(i); 
	    break;
	  }
      }
      in.push_back(xi,v_in);
      out.push_back(xi,v_out);
    }
    

  public: // ---- Named constructors -------------------------------------------------------------------------


    PtensorTransferListObj overlaps(const array_pool<int>& x, const array_pool<int>& y){
      PtensorTransferListObj R;

      if(x.size()<10){
	for(int i=0; i<y.size(); i++){
	  auto v=y(i);
	  for(int j=0; j<x.size(); j++){
	    auto w=x(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      R.append_intersection_of(i,j,x.view_of(i),y.view_of(j));
	  }
	}
	return R;
      }

      map_of_lists2<int,int> in_lists;
      int nx=x.size();
      for(int i=0; i<nx; i++)
	x.for_each_of(i,[&](const int j){in_lists.push_back(j,i);});

      int yx=y.size();
      for(int i=0; i<ny; i++)
	y.for_each_of(i,[&](const int j){
	    for(auto p:in_lists[j])
	      append_intersection_of(p,i,x.view_of(p),y.view_of(i));
	  });
      return R;
    }



  };

}

#endif 
