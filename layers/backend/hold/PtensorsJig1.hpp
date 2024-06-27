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

#ifndef _ptens_PtensorsJig1
#define _ptens_PtensorsJig1

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "RowLevelMap.hpp"
#include "PtensorsJig.hpp"


namespace ptens{


  template<typename DUMMY>
  class PtensorsJig1: public cnine::observable<PtensorsJig1<DUMMY> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef PtensorsJig0<DUMMY> Jig0;
    typedef PtensorsJig1<DUMMY> Jig1;
    typedef PtensorsJig2<DUMMY> Jig2;
    typedef cnine::observable<Jig1> observable;


    shared_ptr<AtomsPackObj> atoms;


    PtensorsJig1(const shared_ptr<AtomsPackObj>& _atoms):
      observable(this),
      atoms(new AtomsPackObj(*_atoms)){} // this copy is to break the circular dependency 

    static shared_ptr<PtensorsJig1<DUMMY> > make_or_cached(const AtomsPack& _atoms){
      return make_or_cached(_atoms.obj);}

    static shared_ptr<PtensorsJig1<DUMMY> > make_or_cached(const shared_ptr<AtomsPackObj>& _atoms){
      if(_atoms->cached_pack1) return _atoms->cached_pack1;
      shared_ptr<PtensorsJig1<DUMMY> > r(new PtensorsJig1(_atoms));
      if(_atoms->cache_packs) _atoms->cached_pack1=r;
      return r;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 1;
    }

    int nrows() const{
      return atoms->nrows1();
    }

    int size_of(const int i) const{
      return atoms->size_of(i);
    }

    int offset(const int i) const{
      return atoms->offset(i);
    }

    int index_of(const int i, const int j0) const{
      return atoms->offset(i)+j0;
    }


  public: // ---- Concatenation -----------------------------------------------------------------------------


    static shared_ptr<Jig1> cat(const vector<Jig1*>& v){
      vector<Jig1*> w;
      for(int i=1; i<v.size(); i++)
	w.push_back(v[i]);
      return v[0]->cat_maps(w);
    }

    typedef cnine::plist_indexed_object_bank<Jig1,shared_ptr<Jig1> > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<Jig1*>& v){
	return shared_ptr<Jig1>(cat_with(v));});

    Jig1* cat_with(const vector<Jig1*>& list){
      cnine::plist<AtomsPackObj*> v;
      for(auto p:list) v.push_back(p->atoms.get());
      return new Jig1(atoms->cat_maps(v));
    }


  public: // ---- Row maps ----------------------------------------------------------------------------------


    RowLevelMap mmap(const AtomsPackMatchObj& lists, const PtensorsJig& y){
      if(dynamic_cast<const Jig0&>(y)) return mmap(lists, dynamic_cast<const Jig0&>(y));
      if(dynamic_cast<const Jig1&>(y)) return mmap(lists, dynamic_cast<const Jig1&>(y));
      if(dynamic_cast<const Jig2&>(y)) return mmap(lists, dynamic_cast<const Jig2&>(y));
      PTENS_UNIMPL();
      return mmap(lists, dynamic_cast<const Jig0&>(y));
    }

    // 1 <- 0
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig0& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=out_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(index_of(out_tensor,out_lists(m,j)),y.index_of(in_tensor));
      }
      
      return cnine::GatherMapProgram(nrows(),y.nrows(),new cnine::GatherMapB(direct));
    }
  

    // 1 <- 1
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig1& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::flog timer("PtensorsJig1::[1<-1]");

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++){
	  direct.push_back(2*index_of(out_tensor,out_lists(m,j))+1,y.index_of(in_tensor,in_lists(m,j)));
	}
      }

      cnine::GatherMapProgram R(nrows(),y.nrows());
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(y.reduce0(in_lists),2,0);
      R.add_map(broadcast0(out_lists,2),1,2);
      R.add_map(new cnine::GatherMapB(direct,2));
      return R;
    }


    // 1 <- 2
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig2& y){
      auto[in_lists,out_lists]=lists.lists();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<in.size(); i0++)
	  direct.push_back(5*index_of(out_tensor,out[i0])+4,y.index_of(in_tensor,in[i0],in[i0]));
	for(int i0=0; i0<in.size(); i0++){
	  for(int i1=0; i1<in.size(); i1++){
	    direct.push_back(5*index_of(out_tensor,out[i0])+3,y.index_of(in_tensor,in[i0],in[i1]));
	    direct.push_back(5*index_of(out_tensor,out[i0])+2,y.index_of(in_tensor,in[i1],in[i0]));
	  }
	}
      }
	
      cnine::GatherMapProgram R;
      R.add_var(Gdims(in_lists.size(),2));
      R.add_map(y.reduce0(in_lists),2,0);
      R.add_map(broadcast0(out_lists,5,0,2),1,2);
      R.add_map(new cnine::GatherMapB(direct,5));
      return R;
    }


  public: // ---- Broadcasting and reduction ----------------------------------------------------------------


    cnine::GatherMapB reduce0(const cnine::hlists<int>& in_lists, const int in_columns=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	in_lists.for_each_of(m,[&](const int x){
	    R.push_back(m,in_columns*index_of(in_tensor,x)+coffs);});
      }
      return cnine::GatherMapB(R,1,in_columns);
    }

    cnine::GatherMapB broadcast0(const cnine::hlists<int>& out_lists, const int stride=1, const int coffs=0, const int out_cols_n=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(stride>=1);
      PTENS_ASSRT(coffs<=stride-1);
      for(int m=0; m<out_lists.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int x){
	    R.push_back(stride*index_of(out_tensor,x)+coffs,m);});
      }
      return cnine::GatherMapB(R,stride,1,out_cols_n);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorsJig1";
    }

    string repr() const{
      return "PtensorsJig1";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const PtensorsJig1<DUMMY>& v){
      stream<<v.str(); return stream;}



  };


  class Jig1ptr: public shared_ptr<PtensorsJig1<int> >{
  public:

    typedef shared_ptr<PtensorsJig1<int> > BASE;

    Jig1ptr(const BASE& x):
      BASE(x){}

    Jig1ptr(const AtomsPack& _atoms):
      BASE(PtensorsJig1<int>::make_or_cached(_atoms)){}

    Jig1ptr(const shared_ptr<AtomsPackObj>& _atoms):
      BASE(PtensorsJig1<int>::make_or_cached(_atoms)){}

  };


}

#endif 


    /*
    template<typename TYPE>
    RowLevelMap rmap(const Ptensors0b<TYPE>& y, const AtomsPackMatch& lists){
      return rmap0(*lists.obj,*y.jig);
    }

    template<typename TYPE>
    RowLevelMap rmap(const Ptensors1b<TYPE>& y, const AtomsPackMatch& lists){
      return rmap1(*lists.obj,*y.jig);
    }

    template<typename TYPE>
    RowLevelMap rmap(const Ptensors2b<TYPE>& y, const AtomsPackMatch& lists){
      return rmap2(*lists.obj,*y.jig);
    }


   typedef cnine::ptr_pair_indexed_object_bank<AtomsPackMatchObj,Jig0,RowLevelMap> MMBank0;
    MMBank0 rmap0=MMBank0([&](const AtomsPackMatchObj& lists, const Jig0& y){
      return mmap(lists,y);});

    typedef cnine::ptr_pair_indexed_object_bank<AtomsPackMatchObj,Jig1,RowLevelMap> MMBank1;
    MMBank1 rmap1=MMBank1([&](const AtomsPackMatchObj& lists, const Jig1& y){
      return mmap(lists,y);});

    typedef cnine::ptr_pair_indexed_object_bank<AtomsPackMatchObj,Jig2,RowLevelMap> MMBank2;
    MMBank2 rmap2=MMBank2([&](const AtomsPackMatchObj& lists, const Jig2& y){
      return mmap(lists,y);});
    */
