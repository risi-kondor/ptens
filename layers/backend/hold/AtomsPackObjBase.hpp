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

#ifndef _ptens_AtomsPackObjBase
#define _ptens_AtomsPackObjBase

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"
#include "ptr_pair_indexed_object_bank.hpp"
#include "observable.hpp"


namespace ptens{


  class AtomsPackObjBase: public cnine::observable<AtomsPackObjBase>{
  public:

    shared_ptr<AtomsPackObj> atoms;

    virtual ~AtomsPackObjBase(){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackObjBase():
      observable(this){}

    AtomsPackObjBase(const shared_ptr<AtomsPackObj>& _atoms):
      observable(this),
      atoms(_atoms){}


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms->size();
    }

    int offset1(const int i) const{
      return atoms->offset(i);
    }

    virtual int size_of(const int i) const=0;
    virtual int offset(const int i) const=0;


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    typedef cnine::ptr_pair_indexed_object_bank<MessageListObj,AtomsPackObjBase,MessageMap> MMBank;
    MMBank message_map=MMBank([&](const MessageListObj& lists, const AtomsPackObjBase& atoms){
      return mmap(lists,atoms);});

    virtual MessageMap mmap(const MessageListObj& lists, const AtomsPackObjBase& y)=0;


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string str(const string indent="") const=0;


  };

}

#endif 
    /*
    MMBank message_map=MMBank([&](const MessageListObj& x){
	if(x.source0.get()){cout<<"k=0"<<endl; return mmap(x,*x.source0);}
	if(x.source1.get()){cout<<"k=1"<<endl; return mmap(x,*x.source1);}
	if(x.source2.get()){cout<<"k=2"<<endl; return mmap(x,*x.source2);}
	CNINE_UNIMPL();
	return mmap(x,*x.source2);
      });
    */
    /*
    MessageMap message_map(const MessageListObj& list, const AtomsPackObjBase& src){
      if(dynamic_cast<const AtomsPack0obj<int>&>(src)) 
	return message_map0(list,dynamic_cast<const AtomsPack0obj<int>&>(src));
      if(dynamic_cast<const AtomsPack1obj<int>&>(src))
	return message_map1(list,dynamic_cast<const AtomsPack1obj<int>&>(src));
      if(dynamic_cast<const AtomsPack2obj<int>&>(src))
	return message_map2(list,dynamic_cast<const AtomsPack2obj<int>&>(src));
    }
    */
    //MMBank message_map0=MMBank([&](const MessageListObj& x, const AtomsPack0obj<int>& src) {return mmap(x,src);});
    //MMBank message_map1=MMBank([&](const MessageListObj& x, const AtomsPack1obj<int>& src) {return mmap(x,src);});
    //MMBank message_map2=MMBank([&](const MessageListObj& x, const AtomsPack2obj<int>& src) {return mmap(x,src);});
     /*
	if(dynamic_cast<AtomsPack0obj<int>*>(&atoms)) 
	  return mmap(lists, dynamic_cast<AtomsPack0obj<int>&>(atoms)); 
	if(dynamic_cast<AtomsPack1obj<int>*>(&atoms)) 
	  return mmap(lists, dynamic_cast<AtomsPack1obj<int>&>(atoms)); 
	if(dynamic_cast<AtomsPack2obj<int>*>(&atoms)) 
	  return mmap(lists, dynamic_cast<AtomsPack2obj<int>&>(atoms)); 
	CNINE_UMINPL();
	return mmap(lists, dynamic_cast<AtomsPack2obj<int>&>(atoms)); 
      });
      */
    /*
    MessageMap mmap(const MessageListObj& lists, const AtomsPackObjBase& y){
      if(dynamic_cast<const AtomsPack0obj<int>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack0obj<int>&>(y));
      if(dynamic_cast<const AtomsPack1obj<int>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack1obj<int>&>(y));
      if(dynamic_cast<const AtomsPack2obj<int>*>(&y)) 
	return mmap(lists, dynamic_cast<const AtomsPack2obj<int>&>(y));
      CNINE_UNIMPL();
      return mmap(lists, dynamic_cast<const AtomsPack0obj<int>&>(y));
    }
    */
  /*
  template<typename ARG0>
  inline MessageMap mmap_dispatch(const ARG0& x, const MessageListObj& lists, const AtomsPackObjBase& y){
    if(dynamic_cast<const AtomsPack0obj<int>*>(&y)) 
      return x.mmap(lists, dynamic_cast<const AtomsPack0obj<int>&>(y));
    if(dynamic_cast<const AtomsPack1obj<int>*>(&y)) 
      return x.mmap(lists, dynamic_cast<const AtomsPack1obj<int>&>(y));
    if(dynamic_cast<const AtomsPack2obj<int>*>(&y)) 
      return x.mmap(lists, dynamic_cast<const AtomsPack2obj<int>&>(y));
    CNINE_UNIMPL();
    return x.mmap(lists, dynamic_cast<const AtomsPack0obj<int>&>(y));
  }
  */

  //template<typename DUMMY> class AtomsPack0obj;
  //template<typename DUMMY> class AtomsPack1obj;
  //template<typename DUMMY> class AtomsPack2obj;
  //class AtomsPackObjBase;

    //public: // ---- Concatenation ------------------------------------------------------------------------------

    /*
    static shared_ptr<AtomsPackObjBase> cat(const vector<shared_ptr<AtomsPackObjBase> >& list){
      CNINE_ASSRT(list.size()>0); 
      cnine::plist<AtomsPackObjBase*> v;
      for(int i=1; i<list.size(); i++)
	v.push_back(list[i].get());
      return list[0]->cat_maps(v);
    }

    typedef cnine::plist_indexed_object_bank<AtomsPackObjBase,shared_ptr<AtomsPackObjBase> > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<AtomsPackObjBase*>& v)
      {return shared_ptr<AtomsPackObjBase>(cat_with(v));});

    virtual AtomsPackObjBase* cat_with(const vector<AtomsPackObjBase*>& list){CNINE_UNIMPL(); return nullptr;};
    */


