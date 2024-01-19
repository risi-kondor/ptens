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

#ifndef _ptens_BatchedAtomsPackNobj
#define _ptens_BatchedAtomsPackNobj

#include "BatchedMessageMap.hpp"
#include "BatchedMessageList.hpp"


namespace ptens{


  template<typename SUB>
  class BatchedAtomsPackNobj: public cnine::object_pack_s<SUB>{
  public:

    typedef cnine::object_pack_s<SUB> BASE;

    //using BASE::BASE;
    using BASE::size;
    using BASE::operator[];

    vector<int> row_offsets;

    BatchedAtomsPackNobj(const vector<shared_ptr<SUB> >& x):
      BASE(x){
      row_offsets.resize(size());
      int t=0;
      if(getk()==0){
	for(int i=0; i<size(); i++){
	  t+=(*this)[i].atoms->tsize0();
	  row_offsets[i]=t;
	}
      }
      if(getk()==1){
	for(int i=0; i<size(); i++){
	  t+=(*this)[i].atoms->tsize1();
	  row_offsets[i]=t;
	}
      }
      if(getk()==2){
	for(int i=0; i<size(); i++){
	  t+=(*this)[i].atoms->tsize2();
	  row_offsets[i]=t;
	}
      }
    }


    //BatchedAtomsPackNobj(const AtomsPackBatch& _atoms){
    //for(auto& p: _atoms.obj)
    //obj.push_back(to_share(new SUB(_atoms)));
    //}


  public: // ---- Access ------------------------------------------------------------------------------------


    int getk() const{
      PTENS_ASSRT(size()>0);
      return (*this)[0].getk();
    }

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

  public: // ---- Message lists ----------------------------------------------------------------------------

    // not cached
    template<typename SRC>
    BatchedMessageList overlaps_mlist(const SRC& x){
      BatchedMessageList R;
      for(int i=0; i<size(); i++)
	R.obj.push_back((*this)[i].atoms->overlaps_mlist(x[i].atoms).obj);
      return R;
    }


  public: // ---- Message maps -----------------------------------------------------------------------------

    // not cached
    template<typename SRC>
    BatchedMessageMap message_map(const BatchedMessageList& lists, const SRC& y){
      cnine::GatherMapProgramPack prog;
      for(int i=0; i<lists.size(); i++)
	prog.obj.push_back((*this)[i]->message_map(lists[i],y[i]));
      return BatchedMessageMap(std::move(prog));
    }

    // not cached
    template<typename SRC>
    BatchedMessageMap inverse_message_map(const BatchedMessageList& lists, const SRC& y){
      cnine::GatherMapProgramPack prog;
      for(int i=0; i<lists.size(); i++)
	prog.obj.push_back((*this)[i]->message_map(lists[i],y[i]).inv());
      return BatchedMessageMap(std::move(prog));
    }

    /*
    BatchedMessageMap message_map(const BatchedMessageList& lists, const BatchedAtomsPackNobj<AtomsPack0obj<int> >& y){
      return mmap0(lists,y);}
    
    BatchedMessageMap message_map(const BatchedMessageList& lists, const BatchedAtomsPackNobj<AtomsPack1obj<int> >& y){
      return mmap1(lists,y);}
    
    BatchedMessageMap message_map(const BatchedMessageList& lists, const BatchedAtomsPackNobj<AtomsPack2obj<int> >& y){
      return mmap2(lists,y);}
    

    typedef cnine::ptr_pair_indexed_object_bank<BatchedMessageList,BatchedAtomsPackNobj<AtomsPack0obj>,BatchedMessageMap> MMBank0;
    MMBank0 mmap0=MMBank0([&](const BatchedMessageList& lists, const BatchedAtomsPackNobj<AtomsPack0obj>& y){
	auto prog=new cnine::GatherMapProgramPack();
	for(int i=0; i<lists.size(); i++)
	  prog->obj.push_back((*this)[i]->message_map(lists[i],y[i]));
	return BatchedMessageMap(prog);
      });

    typedef cnine::ptr_pair_indexed_object_bank<BatchedMessageList,BatchedAtomsPackNobj<AtomsPack1obj>,BatchedMessageMap> MMBank1;
    MMBank1 mmap1=MMBank0([&](const BatchedMessageList& lists, const BatchedAtomsPackNobj<AtomsPack1obj>& y){
	auto prog=new cnine::GatherMapProgramPack();
	for(int i=0; i<lists.size(); i++)
	  prog->obj.push_back((*this)[i]->message_map(lists[i],y[i]));
	return BatchedMessageMap(prog);
      });

    typedef cnine::ptr_pair_indexed_object_bank<BatchedMessageList,BatchedAtomsPackNobj<AtomsPack2obj>,BatchedMessageMap> MMBank2;
    MMBank2 mmap2=MMBank0([&](const BatchedMessageList& lists, const BatchedAtomsPackNobj<AtomsPack2obj>& y){
	auto prog=new cnine::GatherMapProgramPack();
	for(int i=0; i<lists.size(); i++)
	  prog->obj.push_back((*this)[i]->message_map(lists[i],y[i]));
	return BatchedMessageMap(prog);
      });
    */



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack0batchObj";
    }

    string repr() const{
      return "AtomsPack0batchObj";
    }

    //string str(const string indent="") const{
    //return atoms->str(indent);
    //}

    friend ostream& operator<<(ostream& stream, const BatchedAtomsPackNobj& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
