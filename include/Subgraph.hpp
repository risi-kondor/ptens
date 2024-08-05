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

#ifndef _ptens_Subgraph
#define _ptens_Subgraph

#include "PtensSession.hpp"
#include "SubgraphObj.hpp"


namespace ptens{

  //extern PtensSessionObj* ptens_session;



  class Subgraph{
  public:

    //unordered_set<SubgraphObj>::iterator obj;
    shared_ptr<SubgraphObj> obj;
    

  public: // ---- Constructors -------------------------------------------------------------------------------


    Subgraph():
      Subgraph(0,{}){}

    Subgraph(const Subgraph& x):
      obj(x.obj){}

    //Subgraph(unordered_set<SubgraphObj>::iterator _obj):
    //obj(_obj){}

    Subgraph(const shared_ptr<SubgraphObj>& _obj):
      obj(_obj){}

    Subgraph(const int _n):
      obj(ptens_global::subgraph_cache.emplace(_n)){}
    //obj(ptens_global::subgraph_cache.emplace(_n).first){}

    Subgraph(const vector<pair<int,int> >& list): 
      obj(ptens_global::subgraph_cache.emplace(list)){}

    Subgraph(const int _n, const initializer_list<pair<int,int> >& list): 
      obj(ptens_global::subgraph_cache.emplace(_n,list)){}

    Subgraph(const int _n, const initializer_list<pair<int,int> >& list, const cnine::Tensor<float>& labels): 
      obj(ptens_global::subgraph_cache.emplace(_n,list,labels)){}

    Subgraph(const cnine::Tensor<float>& M):
      obj(ptens_global::subgraph_cache.emplace(M)){}

    Subgraph(const cnine::Tensor<float>& M, const cnine::Tensor<float>& L):
      obj(ptens_global::subgraph_cache.emplace(M,L)){}


    static Subgraph edge_index(const cnine::Tensor<int>& x, int _n=-1){
      if(_n==-1) _n=x.max()+1;
      return ptens_global::subgraph_cache.emplace(_n,x);
    }

    static Subgraph edge_index(const cnine::Tensor<int>& x, const cnine::Tensor<int>& L, int _n=-1){
      if(_n==-1) _n=x.max()+1;
      return ptens_global::subgraph_cache.emplace(_n,x,L);
    }

    static Subgraph edge_index_degrees(const cnine::Tensor<int>& x, const cnine::Tensor<int>& D, int _n=-1){
      if(_n==-1) _n=x.max()+1;
      SubgraphObj H(_n,x);
      H.set_degrees(D);
      return ptens_global::subgraph_cache.insert(H);
    }


  public: // ---- Named Constructors -------------------------------------------------------------------------
    

    static Subgraph trivial(){
      return Subgraph(1,{});}

    static Subgraph edge(){
      return Subgraph(2,{{0,1}});}

    static Subgraph triangle(){
      return Subgraph(3,{{0,1},{1,2},{2,0}});}

    static Subgraph cycle(const int n){
      vector<pair<int,int> > v;
      for(int i=0; i<n-1; i++)
	v.push_back(pair<int,int>(i,i+1));
      v.push_back(pair<int,int>(n-1,0));
      return Subgraph(v);
    }

    static Subgraph star(const int n){
      vector<pair<int,int> > v(n-1);
      for(int i=0; i<n-1; i++)
	v[i]=pair<int,int>(0,i+1);
      return Subgraph(v);
    }


  public: // ---- Access --------------------------------------------------------------------------------------


    int getn() const{
      return obj->getn();
    }

    bool has_espaces() const{
      return obj->has_espaces();
    }

    int n_eblocks() const{
      make_eigenbasis();
      return obj->eblocks.size();
    }

    void set_evecs(const cnine::Tensor<float>& _evecs, const cnine::Tensor<float>& _evals) const{
      obj->set_evecs(_evecs,_evals);
    }

    bool operator==(const Subgraph& x) const{
      return &(*obj)==&(*x.obj);
    }

    cnine::Tensor<float> dense() const{
      return obj->dense();
    }

    cnine::Ltensor<float> evecs(){
      make_eigenbasis();
      return obj->evecs;
    }

    void make_eigenbasis() const{
      const_cast<SubgraphObj&>(*obj).make_eigenbasis();
    }

    //bool operator==(const Subgraph& x) const{
    //return obj==x.obj;
    //}


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "Subgraph";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    //static string cached(){
    //ostringstream oss;
    //for(auto p: ptens_global::subgraph_cache)
    //oss<<p<<endl;
    //return oss.str();
    //}

    friend ostream& operator<<(ostream& stream, const Subgraph& x){
      stream<<x.str(); return stream;}

  };


  //class SubgraphCache{};

}


namespace std{
  
  template<>
  struct hash<ptens::Subgraph>{
  public:
    size_t operator()(const ptens::Subgraph& x) const{
      return hash<ptens::SubgraphObj*>()(&const_cast<ptens::SubgraphObj&>(*x.obj));
    }
  };
}


#endif 
