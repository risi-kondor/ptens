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

  template<typename DUMMY>
  class AtomsPack1obj;

  template<typename DUMMY>
  class AtomsPack2obj;


  template<typename DUMMY>
  class AtomsPack0obj: public AtomsPackObjBase{
  public:

    typedef cnine::ptr_indexed_object_bank<MessageListObj,MessageMap> MMBank;


    //shared_ptr<AtomsPackObj> atoms;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack0obj(const int n):
      atoms(new AtomsPackObj(n)){}

    AtomsPack0obj(const AtomsPack& _atoms):
      atoms(_atoms.obj){}

    AtomsPack0obj(const shared_ptr<AtomsPackObj>& _atoms):
      atoms(_atoms){}

    AtomsPack0obj(const initializer_list<initializer_list<int> >& x):
      AtomsPack0obj(cnine::to_share(new AtomsPackObj(x))){}


  public: // ---- Access ------------------------------------------------------------------------------------

/*
    int size() const{
      return atoms->size();
    }

    int offset(const int i) const{
      return i;
    }

    int index_of(const int i) const{
      return i;
    }
*/

  public: // ---- Transfer maps -----------------------------------------------------------------------------


    //template<typename SOURCE>
    //MessageList overlaps_mlist(const SOURCE& x){
    //return MessageList(atoms->overlaps_mlist(x.atoms),x);
    //}


    /*
    MMBank message_map=MMBank([&](const MessageListObj& x){
	if(x.source0) return mmap(x,*x.source0);
	if(x.source1) return mmap(x,*x.source1);
	if(x.source2) return mmap(x,*x.source2);
	CNINE_UNIMPL();
	return mmap(x,*x.source2);
      });
    */

    // 0 <- 0
    MessageMap mmap(const MessageListObj& lists, const AtomsPack0obj<DUMMY>& y){
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
