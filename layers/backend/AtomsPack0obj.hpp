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

#ifndef _ptens_AtomsPack0obj
#define _ptens_AtomsPack0obj

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"
#include "AtomsPackObjBase.hpp"


namespace ptens{

  template<typename DUMMY> class AtomsPack0obj;
  template<typename DUMMY> class AtomsPack1obj;
  template<typename DUMMY> class AtomsPack2obj;


  template<typename DUMMY>
  class AtomsPack0obj: public AtomsPackObjBase{
  public:


    AtomsPack0obj(const int n):
      AtomsPackObjBase(cnine::to_share(new AtomsPackObj(n))){}

    AtomsPack0obj(const AtomsPack& _atoms):
      AtomsPackObjBase(_atoms.obj){}

    AtomsPack0obj(const shared_ptr<AtomsPackObj>& _atoms):
      AtomsPackObjBase(_atoms){}

    AtomsPack0obj(const initializer_list<initializer_list<int> >& x):
      AtomsPack0obj(cnine::to_share(new AtomsPackObj(x))){}
    
    static shared_ptr<AtomsPack0obj<DUMMY> > make_or_cached(const AtomsPack& _atoms){
      return make_or_cached(_atoms.obj);}

    static shared_ptr<AtomsPack0obj<DUMMY> > make_or_cached(const shared_ptr<AtomsPackObj>& _atoms){
      if(_atoms->cached_pack0) return _atoms->cached_pack0;
      shared_ptr<AtomsPack0obj<DUMMY> > r(new AtomsPack0obj(_atoms));
      if(_atoms->cache_packs) _atoms->cached_pack0=r;
      return r;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 0;
    }

    int tsize() const{
      return atoms->tsize0();
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


    typedef cnine::plist_indexed_object_bank<AtomsPackObjBase,shared_ptr<AtomsPack0obj<int> > > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<AtomsPackObjBase*>& v)
      {return shared_ptr<AtomsPack0obj<int> >(cat_with(v));});

    AtomsPack0obj<int>* cat_with(const vector<AtomsPackObjBase*>& list){
      cnine::plist<AtomsPackObj*> v;
      for(auto p:list) v.push_back(p->atoms.get());
      return new AtomsPack0obj<int>(atoms->cat_maps(v));
    }


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    MessageMap mmap(const MessageListObj& lists, const AtomsPackObjBase& y){
      if(dynamic_cast<const AtomsPack0obj<DUMMY>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack0obj<DUMMY>&>(y));
      if(dynamic_cast<const AtomsPack1obj<DUMMY>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack1obj<DUMMY>&>(y));
      if(dynamic_cast<const AtomsPack2obj<DUMMY>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack2obj<DUMMY>&>(y));
      CNINE_UNIMPL();
      return mmap(lists, dynamic_cast<const AtomsPack0obj<DUMMY>&>(y));
    }

    // 0 <- 0
    MessageMap mmap(const MessageListObj& lists, const AtomsPack0obj<DUMMY>& y){
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
    MessageMap mmap(const MessageListObj& lists, const AtomsPack1obj<DUMMY>& y){
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
    MessageMap mmap(const MessageListObj& lists, const AtomsPack2obj<DUMMY>& y){
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
      return "AtomsPack0Obj";
    }

    string repr() const{
      return "AtomsPack0Obj";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack0obj& v){
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
    //auto[in,out]=atoms->overlaps_mlist(*y.atoms).lists();
    //TBANK1 overlaps_map1=TBANK1([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
    //TBANK2 overlaps_map2=TBANK2([&](const AtomsPack0obj<DUMMY>& y){
    //auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();
    //typedef cnine::ptr_indexed_object_bank<MessageListObj,MessageMap> MMBank;


    //shared_ptr<AtomsPackObj> atoms;

