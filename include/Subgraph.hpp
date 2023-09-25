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

#include "SparseRmatrix.hpp"
#include "PtensSession.hpp"


namespace ptens{

  extern PtensSession ptens_session;


  class Subgraph{
  public:

    typedef cnine::SparseRmatrix BaseMatrix;

    unordered_set<SubgraphObj>::iterator obj;
    

  public: // ---- Constructors -------------------------------------------------------------------------------


    Subgraph():
      Subgraph(0,{}){}

    Subgraph(const Subgraph& x):
      obj(x.obj){}

    Subgraph(unordered_set<SubgraphObj>::iterator _obj):
      obj(_obj){}

    Subgraph(const int _n):
      obj(ptens_session.subgraphs.emplace(_n).first){}

    Subgraph(const vector<pair<int,int> >& list): 
      obj(ptens_session.subgraphs.emplace(list).first){}

    Subgraph(const int _n, const initializer_list<pair<int,int> >& list): 
      obj(ptens_session.subgraphs.emplace(_n,list).first){}

    Subgraph(const int _n, const initializer_list<pair<int,int> >& list, const cnine::RtensorA& labels): 
      obj(ptens_session.subgraphs.emplace(_n,list,labels).first){}


    Subgraph(const cnine::RtensorA& M):
      obj(ptens_session.subgraphs.emplace(M).first){}

    Subgraph(const cnine::RtensorA& M, const cnine::RtensorA& L):
      obj(ptens_session.subgraphs.emplace(M,L).first){}




    static Subgraph edge_index(const cnine::RtensorA& x, int _n=-1){
      if(_n==-1) _n=x.max()+1;
      return ptens_session.subgraphs.emplace(_n,x).first;
    }

    static Subgraph edge_index(const cnine::RtensorA& x, const cnine::RtensorA& L, int _n=-1){
      if(_n==-1) _n=x.max()+1;
      return ptens_session.subgraphs.emplace(_n,x,L).first;
    }

    static Subgraph edge_index(const cnine::RtensorA& _edges, const cnine::RtensorA& _evecs, const cnine::RtensorA& _evals){
      int _n=_edges.max()+1;
      return ptens_session.subgraphs.emplace(_n,_edges,_evecs,_evals).first;
    }

    static Subgraph edge_index(const cnine::RtensorA& _edges, const cnine::RtensorA& L, 
      const cnine::RtensorA& _evecs, const cnine::RtensorA& _evals){
      int _n=_edges.max()+1;
      return ptens_session.subgraphs.emplace(_n,_edges,L,_evecs,_evals).first;
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

    int n_eblocks() const{
      make_eigenbasis();
      return obj->eblocks.size();
    }

    void set_evecs(const cnine::TensorView<float>& _evecs, const cnine::TensorView<float>& _evals) const{
      obj->set_evecs(_evecs,_evals);
    }

    bool operator==(const Subgraph& x) const{
      return &(*obj)==&(*x.obj);
    }

    cnine::RtensorA dense() const{
      return obj->dense();
    }

    void make_eigenbasis() const{
      const_cast<SubgraphObj&>(*obj).make_eigenbasis();
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "Subgraph";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Subgraph on "<<obj->getn()<<" vertices:"<<endl;
      oss<<obj->dense().str(indent+"  ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Subgraph& x){
      stream<<x.str(); return stream;}

  };




}

#endif 
