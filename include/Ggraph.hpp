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
 */

#ifndef _Ptens_Graph
#define _Ptens_Graph

#include "Hgraph.hpp"
#include "GgraphObj.hpp"
#include "Subgraph.hpp"
#include "PtensSession.hpp"

extern ptens::PtensSession ptens_session;


namespace ptens{


  class Ggraph{
  public:

    //typedef Hgraph OBJ;
    typedef GgraphObj OBJ;

    shared_ptr<OBJ> obj;

    Ggraph():
      obj(new OBJ()){};

    Ggraph(OBJ* x):
      obj(x){}

    Ggraph(shared_ptr<OBJ>& x):
      obj(x){}

    Ggraph(const shared_ptr<OBJ>& x):
      obj(x){}

    Ggraph(const initializer_list<pair<int,int> >& list, const int n=-1): 
      obj(new OBJ(n,list)){};

    Ggraph(const cnine::RtensorA& M):
      obj(new OBJ(cnine::Tensor<float>(M))){}


    Ggraph(const int key):
      Ggraph(ptens_session.graph_cache(key)){}

    //Ggraph(const Ggraph& x):
    //obj(x.obj){}


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static Ggraph random(const int _n, const float p=0.5){
      return new OBJ(OBJ::random(_n,p));}

    static Ggraph from_edges(const cnine::Tensor<int>& M, const bool cached=false){
      int n=M.max()+1;
      if(!cached) return new OBJ(n,M);
      return ptens_session.graph_cache.from_edge_list(M).second;
    }

    static Ggraph from_edges(int n, const cnine::Tensor<int>& M, const bool cached=false){
      if(n==-1) n=M.max()+1;
      if(!cached) return new OBJ(n,M);
      return ptens_session.graph_cache.from_edge_list(M).second;
    }

    static Ggraph from_edges(const cnine::Tensor<int>& M, const int key){
      int n=M.max()+1;
      return ptens_session.graph_cache.from_edge_list(key,M);
    }

    static Ggraph from_edges(int n, const cnine::Tensor<int>& M, const int key){
      if(n==-1) n=M.max()+1;
      return ptens_session.graph_cache.from_edge_list(key,M);
    }

    static Ggraph from_edge_list(int n, const cnine::Tensor<int>& M){
      if(n==-1) n=M.max()+1;
      return new OBJ(n,M);
    }

    static Ggraph cached_from_edge_list(const cnine::Tensor<int>& M){
      auto [id,G]=ptens_session.graph_cache.from_edge_list(M);
      return Ggraph(G);
    }

    static Ggraph cached_from_edge_list(int n, const cnine::Tensor<int>& M){
      auto [id,G]=ptens_session.graph_cache.from_edge_list(M);
      return Ggraph(G);
    }

    // replace this
    static Ggraph edges(int n, const cnine::RtensorA& M){
      if(n==-1) n=M.max()+1;
      return new OBJ(n,M);
    }


  public: // ---- Access --------------------------------------------------------------------------------------


    int getn() const{
      return obj->getn();
    }

    int nedges() const{
      return obj->nedges();
    }

    AtomsPack edges() const{
      return obj->edges();
    }

    cnine::RtensorA dense() const{
      return obj->dense().rtensor();
    }

    cnine::Ltensor<int> edge_list() const{
      return obj->edge_list();
    }

    void cache(const int key) const{
      ptens_session.graph_cache.cache(key,obj);
    }

    bool operator==(const Ggraph& x) const{
      return obj==x.obj;
    }


  public: // ---- Operations ----------------------------------------------------------------------------------


    Ggraph permute(const cnine::permutation& pi) const{
      return Ggraph(new OBJ(obj->permute(pi)));
    }

    //cnine::array_pool<int> subgraphs_list(const Subgraph& H) const{
    //return obj->subgraphs_list(*H.obj);
    //}

    //cnine::Tensor<int>& subgraphs_matrix(const Subgraph& H) const{
    //return obj->subgraphs_matrix(*H.obj);
    //}

    AtomsPack subgraphs(const Subgraph& H) const{
      return obj->subgraphs(*H.obj);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "ptens::Ggraph";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Ggraph& x){
      stream<<x.str(); return stream;}


  };

}

#endif 


// Ggraph_pack
// MessageListPack
// MessageMapPack
