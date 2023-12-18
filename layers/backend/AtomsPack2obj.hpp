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

  class AtomsPack2obj{
  public:

    shared_ptr<AtomsPackObj> atoms;


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms->size();
    }

    int offset0(const int i){
      return atoms->offset(i);
    }

    int offset(const int i){
      CNINE_UNIMPL();
      return atoms->offset(i);
    }

    int index_of(const int i, const int j0, const int j1){
      CNINE_UNIMPL();
      return atoms->offset(i)+j;
    }


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    CompoundTransferMap overlaps_transfer_map(const AtomsPack0obj<DUMMY>& x){
      return overlaps_transfer_maps0(x);}

    CompoundTransferMap overlaps_transfer_map(const AtomsPack1obj<DUMMY>& x){
      return overlaps_transfer_maps1(x);}

    CompoundTransferMap overlaps_transfer_map(const AtomsPack2obj<DUMMY>& x){
      return overlaps_transfer_maps1(x);}


    // 2 <- 0 
    TBANK0 overlap_transfer_maps0=TBANK0([&](const AtomsPack0obj<DUMMY>& y){
	auto[in_lists,out_lists]=atoms->overlap_lists(*y.atoms)->lists();

	map_of_lists2<int,int> direct;
	for(int m=0; m<in.size(); m++){
	  int in_tensor=in_lists.head(m);
	  int out_tensor=out_lists.head(m);
	  direct.push_back(index_of(out_tensor,out_lists(m,0),out_lists(m,0)),y.index_of(in_tensor));
	}

	return GatherMapProgram R(new GatherMapB(direct));
	});
  
      
    // 2 <- 1
    TBANK1 overlap_transfer_maps1=TBANK1([&](const AtomsPack0obj<DUMMY>& y){
	auto[in_lists,out_lists]=atoms->overlap_lists(*y.atoms)->lists();

	map_of_lists2<int,int> direct;
	for(int m=0; m<in_lists.size(); m++){
	  int in_tensor=in_lists.head(m);
	  int out_tensor=out_lists.head(m);
	  vector<int> in=in_lists(m);
	  vector<int> out=out_lists(m);
	  for(int i0=0; i0<out.size(); i0++){
	    int source=y.index_of(in_tensor,in[i0]);
	    direct.push_back(5*index_of(out_tensor,out[i0],out[i0])+2,source);
	    for(int i1=0; i1<out.size(); i1++){
	      direct.push_back(5*index_of(out_tensor,out[i0],out[i1])+3,source);
	      direct.push_back(5*index_of(out_tensor,out[i1],out[i0])+4,source);
	    }
	  }
	}

	GatherMapProgram R;
	R.add_var(Gdims(in_lists.size(),1));
	R.add_map(new GatherMapB(y.reduce0(in_lists)),2,0);
	R.add_map(new GatherMapB(broadcast0(out_lists,5),5),1,2);
	R.add_map(new GatherMapB(direct,5));
	return R;
      });


    // 2 <- 2
    TBANK2 overlap_transfer_maps2=TBANK2([&](const AtomsPack0obj<DUMMY>& y){
	auto[in,out]=atoms->overlap_lists(*y.atoms)->lists();
	
	map_of_lists2<int,int> direct;
	for(int m=0; m<in.size(); m++){
	  int in_tensor=in.head(m);
	  int out_tensor=out.head(m);
	  vector<int> in=in_lists(m);
	  vector<int> out=out_lists(m);
	  for(int i0=0; i0<in.size_of(m); i0++){
	    for(int i1=0; i1<in.size_of(m); i1++){
	      glists0.push_back(15*index_of(out_tensor,out[i0],out[i1])+13,y.index_of(in_tensor,in[i0],in[i1]));
	      glists0.push_back(15*index_of(out_tensor,out[i0],out[i1])+14,y.index_of(in_tensor,in[i1],in[i0]));
	    }
	  }
	}

	GatherMapProgram R;
	R.add_var(Gdims(in_lists.size(),2));
	R.add_var(Gdims(in_lists.size(),3));

	R.add_map(new GatherMapB(y.reduce0(in_lists),2),2,0);
	R.add_map(new GatherMapB(broadcast0(out_lists,15),15),1,2);

	R.add_map(new GatherMapB(y.reduce1(in_lists),3),3,0);
	R.add_map(new GatherMapB(broadcast1(out_lists,15,4),15),1,3);

	R.add_map(new GatherMapB(direct,15));
	return R;
      });
      

  public: // ---- Broadcasting and reduction ----------------------------------------------------------------


    map_of_lists2<int,int> reduce0(const hlists<int>& in_lists, const int stride=2, const int coffs=0){
      map_of_lists2<int,int> R;
      PTENS_ASSRT(stride>=2);
      PTENS_ASSRT(coffs<=stride-2);

      for(int m=0; m<in_lists.size(); m++){
	
	int in_tensor=in_lists.head(m);
	vector<int> ix=in_lists(m);
	int k=ix.size();
	
	int offs=offset(in_tensor);
	int n=size_of(in_tensor);
	
	for(int i0=0; i0<k; i0++)
	  R.push_back(2*m,stride*(offs+(n+1)*ix[i0])+coffs);
	for(int i=0; i0<k; i0++)
	  for(int i1=0; i1<k; i1++)
	    R.push_back(2*m+1,stride*(offs+ix[i0]*n+ix[i1])+coffs);

      }
      return R;
    }


    map_of_lists2<int,int> reduce1(const hlists<int>& in_lists, const int stride=3, const int coffs=0){
      map_of_lists2<int,int> R;
      PTENS_ASSRT(stride>=3);
      PTENS_ASSRT(coffs<=stride-3);

      for(int m=0; m<in_lists.size(); m++){
	
	int in_tensor=in_lists.head(m);
	vector<int> ix=in_lists(m);
	int k=ix.size();
	
	int in_offs=offset(in_tensor);
	int n=size_of(in_tensor);

	int out_offs=in_lists.offset1(in_tensor); // TODO
	
	for(int i0=0; i0<k; i0++){
	  int target=3*(out_offs+i0);
	  R.push_back(target,stride*(offs+(n+1)*ix[i0])+coffs);
	  for(int i1=0; i1<k; i1++){
	    R.push_back(target+1,stride*(offs+ix[i0]*n+ix[i1])+coffs);
	    R.push_back(target+2,stride*(offs+ix[i1]*n+ix[i0])+coffs);
	  }
	}
      }
      return R;
    }


    map_of_lists2<int,int> broadcast0(const hlists<int>& out_lists, const int stride=2, const int coffs=0){
      map_of_lists2<int,int> R;
      PTENS_ASSRT(stride>=2);
      PTENS_ASSRT(coffs<=stride-2);

      for(int m=0; m<out_lists.size(); m++){
	
	int out_tensor=out_lists.head(m);
	vector<int> ix=out_lists(m);
	int k=ix.size();
	
	int offs=offset(out_tensor);
	int n=size_of(out_tensor);
	
	for(int i0=0; i0<k; i0++)
	  R.push_back(stride*(offs+(n+1)*ix[i0])+coffs,m);
	for(int i=0; i0<k; i0++)
	  for(int i1=0; i1<k; i1++)
	    R.push_back(stride*(offs+ix[i0]*n+ix[i1])+coffs+1,m);

      }
      return R;
    }


    map_of_lists2<int,int> broadcast1(const hlists<int>& out_lists, const int stride=3, const int coffs=0){
      map_of_lists2<int,int> R;
      PTENS_ASSRT(stride>=3);
      PTENS_ASSRT(coffs<=stride-3);

      for(int m=0; m<out_lists.size(); m++){
	
	int out_tensor=out_lists.head(m);
	vector<int> ix=out_lists(m);
	int k=ix.size();
	
	int offs=offset(out_tensor);
	int n=size_of(out_tensor);
	
	int in_offs=out_lists.offset1(in_tensor); // TODO

	for(int i0=0; i0<k; i0++){
	  int source=in_offs+i0;
	  R.push_back(stride*(offs+(n+1)*ix[i0])+coffs,source);
	  for(int i1=0; i1<k; i1++){
	    R.push_back(stride*(offs+ix[i0]*n+ix[i1])+coffs+1,m);
	    R.push_back(stride*(offs+ix[i1]*n+ix[i0])+coffs+2,m);
	  }
	}
      }
      return R;
    }


  };

}

#endif 
	//int source=y.index_of(in_tensor);
	//out_lists.for_each_of(m,[&](const int x){
	//    direct.push_back(2*index_of(out_tensor,x,x),source);});
	//out_lists.for_each_of(m,[&](const int x){
	//    out_lists.for_each_of(m,[&](const int y){
	//	direct.push_back(2*index_of(out_tensor,x,y)+1,source);});
	//});
	
	/*
	map_of_lists2<int,int> reduce0;
	for(int m=0; m<in.size(); m++){
	  int in_tensor=in_lists.head(m);
	  in_lists.for_each_of(m,[&](const int x){
	      reduce0.push_back(m,y.index_of(in_tensor,x));});
	}
      
	map_of_lists2<int,int> broadcast0;
	for(int m=0; m<in.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int p){
	    broadcast0.push_back(5*index_of(out_tensor,p,p),m);
	    out_lists.for_each_of(m,[&](const int q){
		broadcast0.push_back(5*index_of(out_tensor,p,q)+1,m);
	      });
	  });
	}
	*/
	/*
	map_of_lists2<int,int> glists1;
	for(int m=0; m<in.size(); m++){
	  int in_tensor=in.head(m);
	  int out_tensor=out.head(m);
	  for(int i0=0; i0<in.size_of(m); i0++)
	    glists1.push_back(index_of0(out_tensor),index_of(in_tensor,i0));
	}
	
	map_of_lists2<int,int> glists2;
	for(int m=0; m<in.size(); m++){
	  int in_tensor=in.head(m);
	  int out_tensor=out.head(m);
	  for(int i0=0; i0<in.size_of(m); i0++)
	    glists2.push_back(index_of(out_tensor,i0),index_of0(out_tensor));
	}
	*/
