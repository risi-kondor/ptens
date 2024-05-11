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

#ifndef _ptens_MessageList
#define _ptens_MessageList

#include "AtomsPackObj.hpp"
#include "MessageListObj.hpp"
#include "observable.hpp"

namespace ptens{

  //template<typename DUMMY> class AtomsPack0obj;
  //template<typename DUMMY> class AtomsPack1obj;
  //template<typename DUMMY> class AtomsPack2obj;


  class MessageList{
  public:

    shared_ptr<const MessageListObj> obj;

    ~MessageList(){}

    MessageList(){}
    
    MessageList(const MessageListObj* _obj):
      obj(_obj){}


  public: // ---- Named constructors ------------------------------------------------------------------------


    static MessageList overlaps(const cnine::array_pool<int>& in_atoms, 
      const cnine::array_pool<int>& out_atoms){
      return MessageList(new MessageListObj(in_atoms,out_atoms));
    }

    static MessageList overlaps(const cnine::array_pool<int>& in_atoms, 
      const cnine::array_pool<int>& out_atoms, const int min_overlaps){
      return MessageList(new MessageListObj(in_atoms,out_atoms,min_overlaps));
    }

    pair<const cnine::hlists<int>&, const cnine::hlists<int>&> lists() const{
      return pair<const cnine::hlists<int>&, const cnine::hlists<int>&>(obj->in,obj->out);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "MessageList";
    }

    string repr() const{
      return "MessageList";
    }

    string str(const string indent="") const{
      return obj->str();
    }

    friend ostream& operator<<(ostream& stream, const MessageList& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
