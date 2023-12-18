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

#ifndef _ptens_AtomsPackObj1
#define _ptens_AtomsPackObj1

#include "AtomsPackObj.hpp"

namespace ptens{

  class AtomsPack1obj{
  public:

    shared_ptr<AtomsPackObj> atoms;


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms->size();
    }

    int offset(const int i){
      return atoms->offset(i);
    }

    int index_of(const int i, const int j0){
      return atoms->offset(i)+j;
    }


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    CompoundTransferMap overlaps_transfer_map(const AtomsPack0obj<DUMMY>& x){
      return overlaps_transfer_maps0(x);
    }

    CompoundTransferMap overlaps_transfer_map(const AtomsPack1obj<DUMMY>& x){
      return overlaps_transfer_maps1(x);
    }

    CompoundTransferMap overlaps_transfer_map(const AtomsPack2obj<DUMMY>& x){
      return overlaps_transfer_maps1(x);
    }


    // 1 <- 0
    TBANK0 overlap_transfer_maps0=TBANK0([&](const AtomsPack0obj<DUMMY>& y){
	auto[in_lists,out_lists]=atoms->overlap_lists(*y.atoms)->lists();

	map_of_lists2<int,int> direct;
	for(int m=0; m<in_lists.size(); m++){
	  int in_tensor=in_lists.head(m);
	  int out_tensor=out_lists.head(m);
	  direct.push_back(index_of(out_tensor,out_lists(m,0)),y.index_of(in_tensor));
	}
    
	return GatherMapProgram(new GatherMapB(direct));
      });
  

    // 1 <- 1
    TBANK1 overlap_transfer_maps1=TBANK1([&](const AtomsPack0obj<DUMMY>& y){
      auto[in_lists,out_lists]=atoms->overlap_lists(*y.atoms)->lists();

      map_of_lists2<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(2*index_of(out_tensor,out_lists(m,j))+1,y.index_of(in_tensor,in_lists(m,j)));
      }

      GatherMapProgram R;
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(new GatherMapB(y.reduce0(in_lists)),2,0);
      R.add_map(new GatherMapB(broadcast0(out_lists,2),2),1,2);
      R.add_map(new GatherMapB(direct,2));
      return R;
      });


    // 1 <- 2
    TBANK2 overlap_transfer_maps2=TBANK2([&](const AtomsPack0obj<DUMMY>& y){
	auto[in_lists,out_lists]=atoms->overlap_lists(*y.atoms)->lists();

	map_of_lists2<int,int> direct;
	for(int m=0; m<in_lists.size(); m++){
	  int in_tensor=in_lists.head(m);
	  int out_tensor=out_lists.head(m);
	  vector<int> in=in_lists(m);
	  vector<int> out=out_lists(m);
	  for(int i0=0; i0<in.size_of(m); i0++)
	    direct.push_back(5*index_of(out_tensor,out[i0])+2,y.index_of(in_tensor,in[i0],in[i0]));
	  for(int i0=0; i0<in.size_of(m); i0++){
	    for(int i1=0; i1<in.size_of(m); i1++){
	      direct.push_back(5*index_of(out_tensor,out[i0])+3,y.index_of(in_tensor,in[i0],in[i1]));
	      direct.push_back(5*index_of(out_tensor,out[i0])+4,y.index_of(in_tensor,in[i1],in[i0]));
	    }
	  }
	}
	
	GatherMapProgram R;
	R.add_var(Gdims(in_lists.size(),2));
	R.add_map(new GatherMapB(y.reduce0(in_lists),2),2,0);
	R.add_map(new GatherMapB(broadcast0(out_lists,5),5),1,2);
	R.add_map(new GatherMapB(direct,5));
	return R;
      });


  public: // ---- Broadcasting and reduction ----------------------------------------------------------------


    map_of_lists2<int,int> reduce0(const hlists<int>& in_lists, const int stride=1, const int coffs=0){
      map_of_lists2<int,int> R;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	in_lists.for_each_of(m,[&](const int x){
	    R.push_back(m,stride*index_of(in_tensor,x)+coffs);});
      }
      return R;
    }

    map_of_lists2<int,int> broadcast0(const hlists<int>& out_lists, const int stride=1, const int coffs=0){
      map_of_lists2<int,int> R;
      PTENS_ASSRT(stride>=1);
      PTENS_ASSRT(coffs<=stride-1);
      for(int m=0; m<out_lists.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int x){
	    R.push_back(stride*index_of(out_tensor,x)+coffs,m);});
      }
      return R;
    }



  };

}

#endif 
      //map_of_lists2<int,int> reduce0;
      //for(int m=0; m<in.size(); m++){
      //int in_tensor=in_lists.head(m);
      //in_lists.for_each_of(m,[&](const int x){
      //reduce0.push_back(m,y.index_of(in_tensor,x));});
      //}

      //map_of_lists2<int,int> broadcast0;
      //for(int m=0; m<in.size(); m++){
      //int out_tensor=out_lists.head(m);
      //out_lists.for_each_of(m,[&](const int x){
      //broadcast0.push_back(2*index_of(out_tensor,x),m);});
      //}

	/*
	map_of_lists2<int,int> reduce0;
	for(int m=0; m<in.size(); m++){
	int in_tensor=in_lists.head(m);
	in_lists.for_each_of(m,[&](const int x){
	    glists1.push_back(2*m,y.index_of(in_tensor,x,x));});
	in_lists.for_each_of(m,[&](const int p){
	    in.for_each_of(m,[&](const int q){
		glists1.push_back(2*m+1,y.index_of(in_tensor,p,q));});
	  });
	}

	map_of_lists2<int,int> broadcast0;
	for(int m=0; m<in.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int x){
	    broadcast0.push_back(5*index_of(out_tensor,x),m);});
	}
	*/
