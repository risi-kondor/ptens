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

#ifndef _ptens_RowMapFactory0
#define _ptens_RowMapFactory0

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"
#include "AtomsPackObjBase.hpp"


namespace ptens{

  class RowMapFactory0{
  public:

  public: // ---- Transfer maps -----------------------------------------------------------------------------


    // 0 <- 0
    MessageMap mmap(, const AtomsPack0obj<DUMMY>& y, const MessageListObj& lists){
      auto[in,out]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	direct.push_back(index_of(out_tensor),y.index_of(in_tensor));
      }
      return cnine::GatherMapProgram(tsize(),y.tsize(),new cnine::GatherMapB(direct));
    };
  

    // 0 <- 1
    MessageMap mmap(, const AtomsPack1obj<DUMMY>& y, const MessageListObj& lists){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(index_of(out_tensor),y.index_of(in_tensor,in_lists(m,j)));
      }
      return cnine::GatherMapProgram(tsize(),y.tsize(),new cnine::GatherMapB(direct));
    }


    // 0 <- 2
    MessageMap mmap(, const AtomsPack2obj<DUMMY>& y, const MessageListObj& lists){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<in.size(); i0++)
	  direct.push_back(2*index_of(out_tensor)+1,y.index_of(in_tensor,in[i0],in[i0]));
	for(int i0=0; i0<in.size(); i0++)
	  for(int i1=0; i1<in.size(); i1++)
	    direct.push_back(2*index_of(out_tensor),y.index_of(in_tensor,in[i0],in[i1]));
      }
      return cnine::GatherMapProgram(new cnine::GatherMapB(direct,2));
    }

  };

}

#endif
