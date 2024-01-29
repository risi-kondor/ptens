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

#ifndef _ptens_AtomsPack0batchObj
#define _ptens_AtomsPack0batchObj

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"
#include "AtomsPackObjBase.hpp"


namespace ptens{

  template<typename DUMMY> class AtomsPack0batchObj;
  template<typename DUMMY> class AtomsPack1obj;
  template<typename DUMMY> class AtomsPack2obj;


  template<typename DUMMY>
  class AtomsPack0batchObj: public cnine::object_pack_s<AtomsPack0obj>{
  public:

    typedef cnine::object_pack_s<AtomsPack0obj> BASE;

    AtomsPack0batchObj(const AtomsPackBatch& _atoms){
      for(auto& p: _atoms.obj)
	obj.push_back(p);
    }


  public: // ---- Access ------------------------------------------------------------------------------------

    /*
    int offset(const int i) const{
      return i;
    }

    int size_of(const int i) const{
      return atoms->size_of(i);
    }

    int index_of(const int i) const{
      return i;
    }
    */

  public: // ---- Concatenation -----------------------------------------------------------------------------

    /*
    typedef cnine::plist_indexed_object_bank<AtomsPackObjBase,shared_ptr<AtomsPack0batchObj<int> > > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<AtomsPackObjBase*>& v)
      {return shared_ptr<AtomsPack0batchObj<int> >(cat_with(v));});

    AtomsPack0batchObj<int>* cat_with(const vector<AtomsPackObjBase*>& list){
      cnine::plist<AtomsPackObj*> v;
      for(auto p:list) v.push_back(p->atoms.get());
      return new AtomsPack0batchObj<int>(atoms->cat_maps(v));
    }
    */

  public: // ---- Message maps -----------------------------------------------------------------------------


    typedef cnine::ptr_pair_indexed_object_bank<MessageListObj,AtomsPackObjBase,MessageMap> MMBank;
    MMBank message_map0=MMBank([&](const MessageListObj& lists, const AtomsPackObjBase& atoms){
      return mmap(lists,atoms);});


    MessageMap mmap(const MessageListObj& lists, const AtomsPackObjBase& y){
      if(dynamic_cast<const AtomsPack0batchObj<DUMMY>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack0batchObj<DUMMY>&>(y));
      if(dynamic_cast<const AtomsPack1obj<DUMMY>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack1obj<DUMMY>&>(y));
      if(dynamic_cast<const AtomsPack2obj<DUMMY>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack2obj<DUMMY>&>(y));
      CNINE_UNIMPL();
      return mmap(lists, dynamic_cast<const AtomsPack0batchObj<DUMMY>&>(y));
    }

    // 0 <- 0
    MessageMap mmap(const MessageListObj& lists, const AtomsPack0batchObj<DUMMY>& y){
      auto[in,out]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	direct.push_back(index_of(out_tensor),y.index_of(in_tensor));
      }
      return cnine::GatherMapProgram(new cnine::GatherMapB(direct));
    };
  

    // 0 <- 1
    MessageMap mmap(const MessageListObj& lists, const AtomsPack1obj<DUMMY>& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	direct.push_back(index_of(out_tensor),y.index_of(in_tensor,in_lists(m,0)));
      }
      return cnine::GatherMapProgram(new cnine::GatherMapB(direct));
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
      return "AtomsPack0batchObj";
    }

    string repr() const{
      return "AtomsPack0batchObj";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack0batchObj& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
