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

#ifndef _ptens_PtensorsJig0
#define _ptens_PtensorsJig0

#include "map_of_lists.hpp"
#include "AtomsPack.hpp"
#include "AtomsPackObj.hpp"
#include "RowLevelMap.hpp"
#include "PtensorsJig.hpp"


namespace ptens{


  template<typename DUMMY>
  class PtensorsJig0: public cnine::observable<PtensorsJig0<DUMMY> >{
  public:

    typedef PtensorsJig0<DUMMY> Jig0;
    typedef PtensorsJig1<DUMMY> Jig1;
    typedef PtensorsJig2<DUMMY> Jig2;
    typedef cnine::observable<Jig0> observable;


    shared_ptr<AtomsPackObj> atoms;


    PtensorsJig0(const shared_ptr<AtomsPackObj>& _atoms):
      observable(this),
      atoms(new AtomsPackObj(*_atoms)){} // this copy is to break the circular dependency 

    static shared_ptr<PtensorsJig0<DUMMY> > make_or_cached(const AtomsPack& _atoms){
      return make_or_cached(_atoms.obj);}

    static shared_ptr<PtensorsJig0<DUMMY> > make_or_cached(const shared_ptr<AtomsPackObj>& _atoms){
      if(_atoms->cached_pack0) return _atoms->cached_pack0;
      shared_ptr<PtensorsJig0<DUMMY> > r(new PtensorsJig0(_atoms));
      if(_atoms->cache_packs) _atoms->cached_pack0=r;
      return r;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 0;
    }

    int nrows() const{
      return atoms->nrows0();
    }

    int offset(const int i) const{
      return i;
    }

    int size_of(const int i) const{
      return atoms->size_of(i);
    }

    int index_of(const int i) const{
      return i;
    }


  public: // ---- Concatenation -----------------------------------------------------------------------------


    static shared_ptr<Jig0> cat(const vector<Jig0*>& v){
      vector<Jig0*> w;
      for(int i=1; i<v.size(); i++)
	w.push_back(v[i]);
      return v[0]->cat_maps(w);
    }

    typedef cnine::plist_indexed_object_bank<Jig0,shared_ptr<Jig0> > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<Jig0*>& v){
	return shared_ptr<Jig0>(cat_with(v));});

    Jig0* cat_with(const vector<Jig0*>& list){
      cnine::plist<AtomsPackObj*> v;
      for(auto p:list) v.push_back(p->atoms.get());
      return new Jig0(atoms->cat_maps(v));
    }


  public: // ---- Row maps ----------------------------------------------------------------------------------


    RowLevelMap mmap(const AtomsPackMatchObj& lists, const PtensorsJig& y){
      if(dynamic_cast<const Jig0&>(y)) return mmap(lists, dynamic_cast<const Jig0&>(y));
      if(dynamic_cast<const Jig1&>(y)) return mmap(lists, dynamic_cast<const Jig1&>(y));
      if(dynamic_cast<const Jig2&>(y)) return mmap(lists, dynamic_cast<const Jig2&>(y));
      PTENS_UNIMPL();
      return mmap(lists, dynamic_cast<const Jig0&>(y));
    }

    // 0 <- 0
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig0& y){
      auto[in,out]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	direct.push_back(index_of(out_tensor),y.index_of(in_tensor));
      }
      return cnine::GatherMapProgram(nrows(),y.nrows(),new cnine::GatherMapB(direct));
    };
  

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


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorsJig0";
    }

    string repr() const{
      return "PtensorsJig0";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const PtensorsJig0& v){
      stream<<v.str(); return stream;}

  };


  class Jig0ptr: public shared_ptr<PtensorsJig0<int> >{
  public:

    typedef shared_ptr<PtensorsJig0<int> > BASE;

    Jig0ptr(const BASE& x):
      BASE(x){}

    Jig0ptr(const AtomsPack& _atoms):
      BASE(PtensorsJig0<int>::make_or_cached(_atoms)){}

    Jig0ptr(const shared_ptr<AtomsPackObj>& _atoms):
      BASE(PtensorsJig0<int>::make_or_cached(_atoms)){}

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
    */

    /*
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
