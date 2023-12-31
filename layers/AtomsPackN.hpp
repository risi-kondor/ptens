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

#ifndef _ptens_AtomsPackN
#define _ptens_AtomsPackN

#include "AtomsPack.hpp"
#include "AtomsPack0obj.hpp"
#include "AtomsPack1obj.hpp"
#include "AtomsPack2obj.hpp"
#include "AtomsPack0.hpp"
#include "AtomsPack1.hpp"
#include "AtomsPack2.hpp"
#include "MessageMap.hpp"

namespace ptens{

  class AtomsPackN{
  public:


    shared_ptr<AtomsPackObjBase> obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackN(){}

    AtomsPackN(shared_ptr<AtomsPackObjBase> _obj):
      obj(_obj){}

    AtomsPackN(const int n):
      obj(new AtomsPack0obj<int>(n)){}

    AtomsPackN(const int k, const AtomsPack& _atoms){
      if(k==0) obj.reset(new AtomsPack0obj<int>(_atoms));
      if(k==1) obj.reset(new AtomsPack1obj<int>(_atoms));
      if(k==2) obj.reset(new AtomsPack2obj<int>(_atoms));
    }

    AtomsPackN(const int k, const initializer_list<initializer_list<int> >& x){
      if(k==0) obj.reset(new AtomsPack0obj<int>(x));
      if(k==1) obj.reset(new AtomsPack1obj<int>(x));
      if(k==2) obj.reset(new AtomsPack2obj<int>(x));
    }

    //AtomsPackN(AtomsPackObjBase* _obj):
    //obj(_obj){}


    static AtomsPackN cat(const vector<reference_wrapper<AtomsPackN> >& list){
      return AtomsPackObjBase::cat(cnine::mapcar<reference_wrapper<AtomsPackN>,shared_ptr<AtomsPackObjBase> >
	(list,[](const reference_wrapper<AtomsPackN>& x){return x.get().obj;}));}


  public: // ---- Conversions --------------------------------------------------------------------------------


    AtomsPackN(const AtomsPack0& x):
      obj(x.obj){}

    AtomsPackN(const AtomsPack1& x):
      obj(x.obj){}

    AtomsPackN(const AtomsPack2& x):
      obj(x.obj){}


  public: // ---- Maps ---------------------------------------------------------------------------------------

    
    template<typename SOURCE>
    MessageList overlaps_mlist(const SOURCE& x){
      return obj->atoms->overlaps_mlist(*x.obj->atoms);
    }

    template<typename SOURCE>
    MessageMap message_map(const MessageList& list, const SOURCE& source){
      return obj->message_map(*list.obj,*source.obj);
    }

    template<typename SOURCE>
    MessageMap overlaps_mmap(const SOURCE& x){
      return message_map(overlaps_mlist(x),x);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int getk() const{
      if(dynamic_cast<AtomsPack0obj<int>*>(obj.get())) return 0;
      if(dynamic_cast<AtomsPack1obj<int>*>(obj.get())) return 1;
      if(dynamic_cast<AtomsPack2obj<int>*>(obj.get())) return 2;
      CNINE_UNIMPL();
      return 0;
    }

    int size() const{
      return obj->size();
    }

    int size_of(const int i) const{
      return obj->size_of(i);
    }

    int offset(const int i) const{
      return obj->offset(i);
    }

    Atoms operator()(const int i) const{
      return (*obj->atoms)(i);
    }

    int nrows() const{
      if(dynamic_cast<AtomsPack0obj<int>*>(obj.get()))
	return dynamic_cast<AtomsPack0obj<int>*>(obj.get())->size();
      if(dynamic_cast<AtomsPack1obj<int>*>(obj.get()))
	return dynamic_cast<AtomsPack1obj<int>*>(obj.get())->size1();
      if(dynamic_cast<AtomsPack2obj<int>*>(obj.get()))
	return dynamic_cast<AtomsPack2obj<int>*>(obj.get())->size2();
      CNINE_UNIMPL();
      return 0;
    }

    vector<vector<int> > as_vecs() const{
      return obj->atoms->as_vecs();
    }

    bool operator==(const AtomsPackN& y) const{
      return true;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPackN";
    }

    string repr() const{
      return "AtomsPackN";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPackN& v){
      stream<<v.str(); return stream;}



  };

}

#endif 
    /*
    MessageList overlaps_mlist(const AtomsPack0& x){
      return MessageList(obj->atoms->overlaps_mlist(*x.obj->atoms),x.obj);}

    MessageList overlaps_mlist(const AtomsPack1& x){
      return MessageList(obj->atoms->overlaps_mlist(*x.obj->atoms),x.obj);}

    MessageList overlaps_mlist(const AtomsPack2& x){
      return MessageList(obj->atoms->overlaps_mlist(*x.obj->atoms),x.obj);}

    MessageList overlaps_mlist(const AtomsPackN& x){
      MessageList R(obj->atoms->overlaps_mlist(*x.obj->atoms));
      //if(dynamic_cast<AtomsPack0obj<int>*>(x.obj.get())) 
      //R.obj->source0=dynamic_pointer_cast<AtomsPack0obj<int> >(x.obj);
      //if(dynamic_cast<AtomsPack1obj<int>*>(x.obj.get())) 
      //R.obj->source1=dynamic_pointer_cast<AtomsPack1obj<int> >(x.obj);
      //if(dynamic_cast<AtomsPack2obj<int>*>(x.obj.get())) 
      //R.obj->source2=dynamic_pointer_cast<AtomsPack2obj<int> >(x.obj);
      return R;
    }
    */

