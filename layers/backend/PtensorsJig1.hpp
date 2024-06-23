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

#ifndef _ptens_PtensorsJig1
#define _ptens_PtensorsJig1

#include "map_of_lists.hpp"
#include "GatherMapProgram.hpp"
#include "MessageMap.hpp"


namespace ptens{


  template<typename DUMMY>
  class PtensorsJig1: public cnine::observable<PtensorsJig1<DUMMY> >{

  public:

    typedef cnine::Gdims Gdims;
    typedef PtensorsJig0<DUMMY> Jig0;
    typedef PtensorsJig1<DUMMY> Jig1;
    typedef PtensorsJig2<DUMMY> Jig2;
    typedef cnine::observable<Jig1> observable;


    shared_ptr<AtomsPackObj> atoms;


    PtensorsJig1(const shared_ptr<AtomsPackObj>& _atoms):
      observable(this),
      atoms(new AtomsPackObj(*_atoms)){} // this copy is to break the circular dependency 

    static shared_ptr<PtensorsJig1<DUMMY> > make_or_cached(const AtomsPack& _atoms){
      return make_or_cached(_atoms.obj);}

    static shared_ptr<PtensorsJig1<DUMMY> > make_or_cached(const shared_ptr<AtomsPackObj>& _atoms){
      if(_atoms->cached_pack1) return _atoms->cached_pack1;
      shared_ptr<PtensorsJig1<DUMMY> > r(new PtensorsJig1(_atoms));
      if(_atoms->cache_packs) _atoms->cached_pack1=r;
      return r;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 1;
    }

    int nrows() const{
      return atoms->nrows1();
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


  public: // ---- Concatenation -----------------------------------------------------------------------------

  /*
    typedef cnine::plist_indexed_object_bank<PtensorsJig1,shared_ptr<PtensorsJig1<int> > > CAT_MAPS; 
    CAT_MAPS cat_maps=CAT_MAPS([this](const vector<PtensorsJig1*>& v)
      {return shared_ptr<PtensorsJig1<int> >(cat_with(v));});

    PtensorsJig1<int>* cat_with(const vector<PtensorsJig1*>& list){
      cnine::plist<PtensorsJig*> v;
      for(auto p:list) v.push_back(p->atoms.get());
      return new PtensorsJig1<int>(atoms->cat_maps(v));
    }
    */

  public: // ---- Row maps ----------------------------------------------------------------------------------


    template<typename TYPE>
    MessageMap rmap(const Ptensors0b<TYPE>& y, const MessageList& lists){
      return rmap0(*lists.obj,*y.jig);
    }

    template<typename TYPE>
    MessageMap rmap(const Ptensors1b<TYPE>& y, const MessageList& lists){
      return rmap1(*lists.obj,*y.jig);
    }

    template<typename TYPE>
    MessageMap rmap(const Ptensors2b<TYPE>& y, const MessageList& lists){
      return rmap2(*lists.obj,*y.jig);
    }


  private: 

    
   typedef cnine::ptr_pair_indexed_object_bank<MessageListObj,Jig0,MessageMap> MMBank0;
    MMBank0 rmap0=MMBank([&](const MessageListObj& lists, const Jig0& y){
      return mmap(lists,y);});

    typedef cnine::ptr_pair_indexed_object_bank<MessageListObj,Jig1,MessageMap> MMBank1;
    MMBank1 rmap1=MMBank([&](const MessageListObj& lists, const Jig1& y){
      return mmap(lists,y);});

    typedef cnine::ptr_pair_indexed_object_bank<MessageListObj,Jig2,MessageMap> MMBank2;
    MMBank2 rmap2=MMBank([&](const MessageListObj& lists, const Jig2& y){
      return mmap(lists,y);});


  private:

    // 1 <- 0
    MessageMap mmap(const MessageListObj& lists, const Jig0& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=out_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(index_of(out_tensor,out_lists(m,j)),y.index_of(in_tensor));
      }
      
      return cnine::GatherMapProgram(nrows(),y.nrows(),new cnine::GatherMapB(direct));
    }
  

    // 1 <- 1
    MessageMap mmap(const MessageListObj& lists, const Jig1& y){
      auto[in_lists,out_lists]=lists.lists();
      cnine::flog timer("PtensorsJig1::[1<-1]");

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++){
	  direct.push_back(2*index_of(out_tensor,out_lists(m,j))+1,y.index_of(in_tensor,in_lists(m,j)));
	}
      }

      cnine::GatherMapProgram R(nrows(),y.nrows());
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(y.reduce0(in_lists),2,0);
      R.add_map(broadcast0(out_lists,2),1,2);
      R.add_map(new cnine::GatherMapB(direct,2));
      return R;
    }


    // 1 <- 2
    MessageMap mmap(const MessageListObj& lists, const Jig2& y){
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
      R.add_map(broadcast0(out_lists,5,0,2),1,2);
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

    cnine::GatherMapB broadcast0(const cnine::hlists<int>& out_lists, const int stride=1, const int coffs=0, const int out_cols_n=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(stride>=1);
      PTENS_ASSRT(coffs<=stride-1);
      for(int m=0; m<out_lists.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int x){
	    R.push_back(stride*index_of(out_tensor,x)+coffs,m);});
      }
      return cnine::GatherMapB(R,stride,1,out_cols_n);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorsJig1";
    }

    string repr() const{
      return "PtensorsJig1";
    }

    string str(const string indent="") const{
      return atoms->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const PtensorsJig1<DUMMY>& v){
      stream<<v.str(); return stream;}



  };

}

#endif 

