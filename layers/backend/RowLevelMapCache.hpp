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
 */

#ifndef _ptens_RowLevelMapCache
#define _ptens_RowLevelMapCache

#include "ptr_triple_indexed_cache.hpp"
//#include "AtomsPack.hpp"
//#include "AtomsPackObj.hpp"
#include "AtomsPackTag.hpp"
#include "TensorLevelMapObj.hpp"
#include "RowLevelMap.hpp"


namespace ptens{

  typedef AtomsPackObj DUMMYC;


  namespace ptens_global{
    extern bool cache_rmaps;
  }


  class RowLevelMapCache: 
    public cnine::ptr_triple_indexed_cache<AtomsPackTagObj,AtomsPackTagObj,TensorLevelMapObj,shared_ptr<RowLevelMap> >{
  public:

    typedef std::tuple<AtomsPackTagObj*,AtomsPackTagObj*,TensorLevelMapObj*> KEYS;
    typedef shared_ptr<RowLevelMap> OBJ;
    typedef cnine::ptr_triple_indexed_cache<AtomsPackTagObj,AtomsPackTagObj,TensorLevelMapObj,shared_ptr<RowLevelMap> > BASE;

    RowLevelMapCache(){}


  public: // ---- Access ------------------------------------------------------------------------------------------


    //shared_ptr<RowLevelMap> operator()(const AtomsPackTagObj& out, const AtomsPackTagObj& in, const shared_ptr<TensorLevelMapObj>){
      //if(ptens_global::cache_row_level_mmaps) return BASE::operator()(out,in); 
      //return shared_ptr<TensorLevelMapObj>(new TensorLevelMapObj(in,out));
      //return shared_ptr<TensorLevelMapObj>(new TensorLevelMapObj());
    //}

    //shared_ptr<RowLevelMap> operator()(const AtomsPackTag0& out, const AtomsPackTag0& in, const shared_ptr<TensorLevelMapObj>& map){

    template<typename OUT_TAG, typename IN_TAG>
    shared_ptr<RowLevelMap> operator()(const OUT_TAG& out, const IN_TAG& in, const shared_ptr<TensorLevelMapObj>& map){
      auto out_p=out.obj.get();
      auto in_p=in.obj.get();
      auto p=make_tuple(out_p,in_p,map.get());
      auto it=find(p);
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      auto r=mmap(*out.obj,*in.obj,*map);
      BASE::insert(out_p,in_p,map.get(),r);
      return r;
      //return OBJ(new RowLevelMap());
    }
    

  private: // ---- Zeroth order ------------------------------------------------------------------------------------


    // 0 <- 0
    shared_ptr<RowLevelMap> mmap(const AtomsPackTagObj0& _x, const AtomsPackTagObj0& _y, const TensorLevelMapObj& map){
      auto x=_x.get_atoms();
      auto y=_y.get_atoms();
      auto[in,out]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	direct.push_back(x.index_of0(out_tensor),y.index_of0(in_tensor));
      }
      return shared_ptr<RowLevelMap>
	(new RowLevelMap(cnine::GatherMapProgram(x.nrows0(),y.nrows0(),new cnine::GatherMapB(direct))));
    };
  

    /*
    // 0 <- 1
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig1& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(index_of(out_tensor),y.index_of(in_tensor,in_lists(m,j)));
      }
      return cnine::GatherMapProgram(nrows(),y.nrows(),new cnine::GatherMapB(direct));
    }


    // 0 <- 2
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig2& y){
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
    */

  };

}

#endif 
