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

#ifndef _ptens_AtomsPackObj2
#define _ptens_AtomsPackObj2

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "AtomsPack0obj.hpp"
#include "AtomsPack1obj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"


namespace ptens{

  template<typename DUMMY>
  class AtomsPack0obj;

  template<typename DUMMY>
  class AtomsPack1obj;


  template<typename DUMMY>
  class AtomsPack2obj{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::ptr_indexed_object_bank<MessageListObj,MessageMap> MMBank;


    shared_ptr<AtomsPackObj> atoms;
    vector<int> offsets;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack2obj(const shared_ptr<AtomsPackObj>& _atoms):
      atoms(_atoms),
      offsets(_atoms->size()){
      int t=0;
      for(int i=0; i<atoms->size(); i++){
	offsets[i]=t;
	t+=pow(atoms->size_of(i),2);
      }
    }

    AtomsPack2obj(const AtomsPack& _atoms):
      AtomsPack2obj(_atoms.obj){}

    AtomsPack2obj(const initializer_list<initializer_list<int> >& x):
      AtomsPack2obj(cnine::to_share(new AtomsPackObj(x))){}


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms->size();
    }

    int size1() const{
      return atoms->tsize1();
    }

    int size2() const{
      return atoms->tsize2();
    }

    //int offset1(const int i) const{
    //return atoms->offset(i);
    //}

    int size_of(const int i) const{
      return atoms->size_of(i);
    }

    int offset(const int i) const{
      return offsets[i];
    }

    int index_of(const int i, const int j0, const int j1) const{
      return offsets[i]+j0*atoms->size_of(i)+j1;
    }


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    MMBank message_map=MMBank([&](const MessageListObj& x){
	if(x.source0) return mmap(x,*x.source0);
	if(x.source1) return mmap(x,*x.source1);
	if(x.source2) return mmap(x,*x.source2);
	CNINE_UNIMPL();
	return mmap(x,*x.source2);
      });


    // 2 <- 0 
    MessageMap mmap(const MessageListObj& lists, const AtomsPack0obj<DUMMY>& y){
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
    MessageMap mmap(const MessageListObj& lists, const AtomsPack1obj<DUMMY>& y){
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
    MessageMap mmap(const MessageListObj& lists, const AtomsPack2obj<DUMMY>& y){
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


    cnine::GatherMapB broadcast0(const cnine::hlists<int>& out_lists, const int ncols=2, const int coffs=0, const int cstride=1) const{
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
      return cnine::GatherMapB(R,ncols);
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
      return cnine::GatherMapB(R,ncols);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack2obj";
    }

    string repr() const{
      return "AtomsPack2obj";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack2obj& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
    //GatherMapProgram overlaps_map(const AtomsPack0obj<DUMMY>& x){
    //return overlaps_map0(x);}

    //GatherMapProgram overlaps_map(const AtomsPack1obj<DUMMY>& x){
    //return overlaps_map1(x);}

    //GatherMapProgram overlaps_map(const AtomsPack2obj<DUMMY>& x){
    //return overlaps_map2(x);}

    //typedef cnine::ptr_indexed_object_bank<AtomsPack0obj<DUMMY>,GatherMapProgram> TBANK0;
    //typedef cnine::ptr_indexed_object_bank<AtomsPack1obj<DUMMY>,GatherMapProgram> TBANK1;
    //typedef cnine::ptr_indexed_object_bank<AtomsPack2obj<DUMMY>,GatherMapProgram> TBANK2;
    //TBANK0 overlaps_map0=TBANK0([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
    //TBANK1 overlaps_map1=TBANK1([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
    //TBANK2 overlaps_map2=TBANK2([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in,out]=atoms->overlaps_mlist(*y.atoms).lists();
