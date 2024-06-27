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

#ifndef _ptens_PtensorsJig2
#define _ptens_PtensorsJig2

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "RowLevelMap.hpp"
#include "PtensorsJig.hpp"


namespace ptens{


  template<typename DUMMY>
  class PtensorsJig2: public cnine::observable<PtensorsJig2<DUMMY> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef PtensorsJig0<DUMMY> Jig0;
    typedef PtensorsJig1<DUMMY> Jig1;
    typedef PtensorsJig2<DUMMY> Jig2;
    typedef cnine::observable<Jig2> observable;

    shared_ptr<AtomsPackObj> atoms;
    vector<int> offsets;


  public: // ---- Constructors ------------------------------------------------------------------------------


    PtensorsJig2(const shared_ptr<AtomsPackObj>& _atoms):
      observable(this),
      atoms(new AtomsPackObj(*_atoms)), // this copy is to break the circular dependency 
      offsets(_atoms->size()){
      int t=0;
      for(int i=0; i<atoms->size(); i++){
	offsets[i]=t;
	t+=pow(atoms->size_of(i),2);
      }
    }

    static shared_ptr<PtensorsJig2<DUMMY> > make_or_cached(const AtomsPack& _atoms){
      return make_or_cached(_atoms.obj);}

    static shared_ptr<PtensorsJig2<DUMMY> > make_or_cached(const shared_ptr<AtomsPackObj>& _atoms){
      if(_atoms->cached_pack2) return _atoms->cached_pack2;
      shared_ptr<PtensorsJig2<DUMMY> > r(new PtensorsJig2(_atoms));
      if(_atoms->cache_packs) _atoms->cached_pack2=r;
      return r;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 2;
    }

    int nrows() const{
      return atoms->nrows2();
    }

    int offset1(const int i) const{
      return atoms->row_offset1(i); // changed
    }

    int size_of(const int i) const{
      return atoms->size_of(i);
    }

    int offset(const int i) const{
      return offsets[i];
    }

    int index_of(const int i, const int j0, const int j1) const{
      return offsets[i]+j0*atoms->size_of(i)+j1;
    }


  public: // ---- Concatenation -----------------------------------------------------------------------------


    /*
    typedef cnine::plist_indexed_object_bank<AtomsPackObjBase,shared_ptr<PtensorsJig2<int> > > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<AtomsPackObjBase*>& v)
      {return shared_ptr<PtensorsJig2<int> >(cat_with(v));});

    PtensorsJig2<int>* cat_with(const vector<AtomsPackObjBase*>& list){
      cnine::plist<AtomsPackObj*> v;
      for(auto p:list) v.push_back(p->atoms.get());
      return new PtensorsJig2<int>(atoms->cat_maps(v));
    }
    */


  public: // ---- Row maps ----------------------------------------------------------------------------------


    RowLevelMap mmap(const AtomsPackMatchObj& lists, const PtensorsJig& y){
      if(dynamic_cast<const Jig0&>(y)) return mmap(lists, dynamic_cast<const Jig0&>(y));
      if(dynamic_cast<const Jig1&>(y)) return mmap(lists, dynamic_cast<const Jig1&>(y));
      if(dynamic_cast<const Jig2&>(y)) return mmap(lists, dynamic_cast<const Jig2&>(y));
      PTENS_UNIMPL();
      return mmap(lists, dynamic_cast<const Jig0&>(y));
    }

    // 2 <- 0 
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig0& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<out.size(); i0++)
	  direct.push_back(2*index_of(out_tensor,out[i0],out[0])+1,y.index_of(in_tensor));
	for(int i0=0; i0<out.size(); i0++)
	  for(int i1=0; i1<out.size(); i1++)
	    direct.push_back(2*index_of(out_tensor,out[i0],out[i1]),y.index_of(in_tensor));
      }
      return cnine::GatherMapProgram(new cnine::GatherMapB(direct,2));
    }
  
      
    // 2 <- 1
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig1& y){
      auto[in_lists,out_lists]=lists.lists();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<out.size(); i0++){
	  int source=y.index_of(in_tensor,in[i0]);
	  direct.push_back(5*index_of(out_tensor,out[i0],out[i0])+4,source);
	  for(int i1=0; i1<out.size(); i1++){
	    direct.push_back(5*index_of(out_tensor,out[i0],out[i1])+3,source);
	    direct.push_back(5*index_of(out_tensor,out[i1],out[i0])+2,source);
	  }
	}
      }

      cnine::GatherMapProgram R;
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(y.reduce0(in_lists),2,0);
      R.add_map(broadcast0(out_lists,5),1,2);
      R.add_map(new cnine::GatherMapB(direct,5));
      return R;
    }


    // 2 <- 2
    RowLevelMap mmap(const AtomsPackMatchObj& lists, const Jig2& y){
      auto[in_lists,out_lists]=lists.lists();
	
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<in.size(); i0++){
	  for(int i1=0; i1<in.size(); i1++){
	    direct.push_back(15*index_of(out_tensor,out[i0],out[i1])+13,y.index_of(in_tensor,in[i0],in[i1]));
	    direct.push_back(15*index_of(out_tensor,out[i0],out[i1])+14,y.index_of(in_tensor,in[i1],in[i0]));
	  }
	}
      }

      cnine::GatherMapProgram R;
      R.add_var(Gdims(in_lists.size(),2));
      R.add_map(y.reduce0(in_lists),2,0);
      R.add_map(broadcast0(out_lists,15,0,2),1,2);

      R.add_var(Gdims(in_lists.get_tail()-in_lists.size(),3));
      R.add_map(y.reduce1(in_lists),3,0);
      R.add_map(broadcast1(out_lists,15,4,3),1,3);

      R.add_map(new cnine::GatherMapB(direct,15));
      return R;
    }
      

  public: // ---- Broadcasting and reduction ----------------------------------------------------------------


    cnine::GatherMapB reduce0(const cnine::hlists<int>& in_lists, const int in_columns=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(coffs<=in_columns-1);

      for(int m=0; m<in_lists.size(); m++){
	
	int in_tensor=in_lists.head(m);
	vector<int> ix=in_lists(m);
	int k=ix.size();
	
	int offs=offset(in_tensor);
	int n=size_of(in_tensor);
	
	for(int i0=0; i0<k; i0++)
	  R.push_back(2*m+1,in_columns*(offs+(n+1)*ix[i0])+coffs);
	for(int i0=0; i0<k; i0++)
	  for(int i1=0; i1<k; i1++)
	    R.push_back(2*m,in_columns*(offs+ix[i0]*n+ix[i1])+coffs);

      }
      return cnine::GatherMapB(R,2,in_columns);
    }


    cnine::GatherMapB reduce1(const cnine::hlists<int>& in_lists, const int in_columns=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(coffs<=in_columns-1);

      for(int m=0; m<in_lists.size(); m++){
	
	int in_tensor=in_lists.head(m);
	vector<int> ix=in_lists(m);
	int k=ix.size();
	
	int offs=/*atoms->*/offset(in_tensor);
	int n=size_of(in_tensor);

	int out_offs=in_lists.offset(m)-m; 
	
	for(int i0=0; i0<k; i0++){
	  int target=3*(out_offs+i0);
	  R.push_back(target+2,in_columns*(offs+(n+1)*ix[i0])+coffs);
	  for(int i1=0; i1<k; i1++){
	    R.push_back(target+1,in_columns*(offs+ix[i0]*n+ix[i1])+coffs);
	    R.push_back(target,in_columns*(offs+ix[i1]*n+ix[i0])+coffs);
	  }
	}
      }
      return cnine::GatherMapB(R,3,in_columns);
    }


    cnine::GatherMapB broadcast0(const cnine::hlists<int>& out_lists, const int ncols=2, const int coffs=0, 
      const int cstride=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(ncols>=2);
      PTENS_ASSRT(coffs<=ncols-2);

      for(int m=0; m<out_lists.size(); m++){
	
	int out_tensor=out_lists.head(m);
	vector<int> ix=out_lists(m);
	int k=ix.size();
	
	int offs=offset(out_tensor);
	int n=size_of(out_tensor);
	
	for(int i0=0; i0<k; i0++)
	  R.push_back(ncols*(offs+(n+1)*ix[i0])+coffs+cstride,m);
	for(int i0=0; i0<k; i0++)
	  for(int i1=0; i1<k; i1++)
	    R.push_back(ncols*(offs+ix[i0]*n+ix[i1])+coffs,m);

      }
      return cnine::GatherMapB(R,ncols,1,cstride);
    }


    cnine::GatherMapB broadcast1(const cnine::hlists<int>& out_lists, const int ncols=3, const int coffs=0, const int cstride=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(ncols>=3);
      PTENS_ASSRT(coffs<=ncols-3);

      for(int m=0; m<out_lists.size(); m++){
	
	int out_tensor=out_lists.head(m);
	vector<int> ix=out_lists(m);
	int k=ix.size();
	
	int offs=offset(out_tensor);
	int n=size_of(out_tensor);
	
	int in_offs=out_lists.offset(m)-m;

	for(int i0=0; i0<k; i0++){
	  int source=in_offs+i0;
	  R.push_back(ncols*(offs+(n+1)*ix[i0])+coffs+2*cstride,source);
	  for(int i1=0; i1<k; i1++){
	    R.push_back(ncols*(offs+ix[i0]*n+ix[i1])+coffs+cstride,source);
	    R.push_back(ncols*(offs+ix[i1]*n+ix[i0])+coffs,source);
	  }
	}
      }
      return cnine::GatherMapB(R,ncols,1,cstride);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorsJig2";
    }

    string repr() const{
      return "PtensorsJig2";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Jig2& v){
      stream<<v.str(); return stream;}


  };


  class Jig2ptr: public shared_ptr<PtensorsJig2<int> >{
  public:

    typedef shared_ptr<PtensorsJig2<int> > BASE;

    Jig2ptr(const BASE& x):
      BASE(x){}

    Jig2ptr(const AtomsPack& _atoms):
      BASE(PtensorsJig2<int>::make_or_cached(_atoms)){}

    Jig2ptr(const shared_ptr<AtomsPackObj>& _atoms):
      BASE(PtensorsJig2<int>::make_or_cached(_atoms)){}

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
