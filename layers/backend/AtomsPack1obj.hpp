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

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "AtomsPack0obj.hpp"
#include "AtomsPack2obj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"


namespace ptens{

  template<typename DUMMY> class AtomsPack0obj;
  template<typename DUMMY> class AtomsPack2obj;


  template<typename DUMMY>
  class AtomsPack1obj{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::ptr_indexed_object_bank<MessageListObj,MessageMap> MMBank;

    shared_ptr<AtomsPackObj> atoms;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack1obj(const AtomsPack& _atoms):
      atoms(_atoms.obj){}

    AtomsPack1obj(const shared_ptr<AtomsPackObj>& _atoms):
      atoms(_atoms){}

    AtomsPack1obj(const initializer_list<initializer_list<int> >& x):
      AtomsPack1obj(cnine::to_share(new AtomsPackObj(x))){}


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms->size();
    }

    int size1() const{
      return atoms->tsize1();
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


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    MMBank message_map=MMBank([&](const MessageListObj& x){
	if(x.source0) return mmap(x,*x.source0);
	if(x.source1) return mmap(x,*x.source1);
	if(x.source2) return mmap(x,*x.source2);
	CNINE_UNIMPL();
	return mmap(x,*x.source2);
      });


    // 1 <- 0
    MessageMap mmap(const MessageListObj& lists, const AtomsPack0obj<DUMMY>& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	direct.push_back(index_of(out_tensor,out_lists(m,0)),y.index_of(in_tensor));
      }
      
      return cnine::GatherMapProgram(new cnine::GatherMapB(direct));
    }
  

    // 1 <- 1
    MessageMap mmap(const MessageListObj& lists, const AtomsPack1obj<DUMMY>& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::flog timer("AtomsPack1obj::[1<-1]");

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(2*index_of(out_tensor,out_lists(m,j))+1,y.index_of(in_tensor,in_lists(m,j)));
      }

      cnine::GatherMapProgram R;
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(y.reduce0(in_lists),2,0);
      R.add_map(broadcast0(out_lists,2),1,2);
      R.add_map(new cnine::GatherMapB(direct,2));
      return R;
    }


    // 1 <- 2
    MessageMap mmap(const MessageListObj& lists, const AtomsPack2obj<DUMMY>& y){
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
      R.add_map(broadcast0(out_lists,5),1,2);
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

    cnine::GatherMapB broadcast0(const cnine::hlists<int>& out_lists, const int stride=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(stride>=1);
      PTENS_ASSRT(coffs<=stride-1);
      for(int m=0; m<out_lists.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int x){
	    R.push_back(stride*index_of(out_tensor,x)+coffs,m);});
      }
      return cnine::GatherMapB(R,stride);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack1obj";
    }

    string repr() const{
      return "AtomsPack1obj";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack1obj& v){
      stream<<v.str(); return stream;}



  };

}

#endif 


    //GatherMapProgram overlaps_map(const AtomsPack0obj<DUMMY>& x){
    //return overlaps_map0(x);}

    //GatherMapProgram overlaps_map(const AtomsPack1obj<DUMMY>& x){
    //return overlaps_map1(x);}

    //GatherMapProgram overlaps_map(const AtomsPack2obj<DUMMY>& x){
    //return overlap_map2(x);}
    //typedef cnine::ptr_indexed_object_bank<AtomsPack0obj<DUMMY>,GatherMapProgram> TBANK0;
    //typedef cnine::ptr_indexed_object_bank<AtomsPack1obj<DUMMY>,GatherMapProgram> TBANK1;
    //typedef cnine::ptr_indexed_object_bank<AtomsPack2obj<DUMMY>,GatherMapProgram> TBANK2;
    //TBANK0 overlaps_map0=TBANK0([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
    //TBANK1 overlaps_map1=TBANK1([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
    //TBANK2 overlaps_map2=TBANK2([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
