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


namespace ptens{


  class Ggraph{
  public:

    //typedef Hgraph BASE;
    typedef GgraphObj BASE;

    shared_ptr<BASE> obj;

    Ggraph():
      obj(new BASE()){};

    Ggraph(BASE* x):
      obj(x){}

    Ggraph(const initializer_list<pair<int,int> >& list, const int n=-1): 
      obj(new BASE(n,list)){};

    Ggraph(const cnine::RtensorA& M):
      obj(new BASE(cnine::Tensor<float>(M))){}


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static Ggraph random(const int _n, const float p=0.5){
      return new BASE(BASE::random(_n,p));}

    static Ggraph from_edges(int n, const cnine::Tensor<int>& M){
      if(n==-1) n=M.max()+1;
      return new BASE(n,M);
    }

    // replace this
    static Ggraph edges(int n, const cnine::RtensorA& M){
      if(n==-1) n=M.max()+1;
      return new BASE(n,M);
    }


  public: // ---- Access --------------------------------------------------------------------------------------


    int getn() const{
      return obj->getn();
    }

    cnine::RtensorA dense() const{
      return obj->dense().rtensor();
    }

    bool operator==(const Ggraph& x) const{
      return obj==x.obj;
    }


  public: // ---- Operations ----------------------------------------------------------------------------------


    Ggraph permute(const cnine::permutation& pi) const{
      return Ggraph(new BASE(obj->permute(pi)));
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
