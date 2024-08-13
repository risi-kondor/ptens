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

#ifndef _ptens_BatchedAindexPackB
#define _ptens_BatchedAindexPackB

#include "object_pack_s.hpp"
#include "AindexPackB.hpp"

namespace ptens{


  class BatchedAindexPackB: public cnine::object_pack_s<AindexPackB>{
  public:

    typedef cnine::object_pack_s<AindexPackB> BASE;
    typedef cnine::Ltensor<int> ITENSOR;

    using BASE::BASE;

    int nrows=0;
    int count1=0;
    int count2=0;

    cnine::RemoteCopy<int,ITENSOR> on_device=cnine::RemoteCopy<int,ITENSOR>([this](const int& _dev){
	auto p=fuse_on_device(_dev);
	gmap_on_device.insert(_dev,p.second);
	return p.first;
      });

    cnine::RemoteCopy<int,ITENSOR> gmap_on_device=cnine::RemoteCopy<int,ITENSOR>([this](const int& _dev){
	auto p=fuse_on_device(_dev);
	gmap_on_device.insert(_dev,p.first);
	return p.second;
      });

 

  public: // -------------------------------------------------------------------------------------------------


    pair<shared_ptr<ITENSOR>,shared_ptr<ITENSOR> > fuse_on_device(const int dev){

      int N=size();
      int nrows=0;
      int max_width=0;
      for(auto& p: obj){
	nrows+=p->size();
	cnine::bump(max_width,p->dim(1));
      }
      ITENSOR* ipack=new ITENSOR({nrows,max_width},1,dev);

      int tail=0;
      int xoffs=0;
      int roffs=0;
      for(auto& p: obj){
	auto M=p->on_device(dev);
	if(M.dim(1)==max_width) ipack->rows(tail,M.dim(0))+=M;
	else ipack->block(tail,0,M.dim(0),M.dim(1))+=M;
	ipack->rows(tail,M.dim(0)).col(0)+=roffs;
	ipack->rows(tail,M.dim(0)).col(2)+=xoffs;
	tail+=M.dim(0);
	roffs+=p->nrows;
	xoffs+=p->n_input_rows;
      }

      int gmap_total=0;
      int n_gather_lists=0;
      for(auto& p: obj){
	auto M=p->gmap_on_device(dev);
	gmap_total+=M.dim(0);
	n_gather_lists+=p->n_gather_lists;
      }
      ITENSOR* gmap=new ITENSOR({gmap_total-2*N+2},1,dev);

      gmap->set(0,n_gather_lists);
      int index_tail=1;
      int data_tail=n_gather_lists+2;
      for(auto& p: obj){
	auto M=p->gmap_on_device(dev);

	int n_lists=p->n_gather_lists;
	gmap->block(index_tail,n_lists)=M.block(1,n_lists);
	gmap->block(index_tail,n_lists)+=data_tail-(n_lists+2);

	int n_data=M.dim(0)-n_lists-2;
	gmap->block(data_tail,n_data)=M.block(n_lists+2,n_data);
	gmap->block(data_tail,n_data)+=p->dim(0);

	index_tail+=n_lists;
	data_tail+=n_data;
      }
      gmap->set(n_gather_lists+1,data_tail);
      
      return make_pair(to_share(ipack),to_share(gmap));
    }
    

  };

}


#endif 
