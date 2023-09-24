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


namespace ptens{


  class Ggraph{
  public:

    shared_ptr<Hgraph> obj;

    Ggraph():
      obj(new Hgraph()){};

    Ggraph(Hgraph* x):
      obj(x){}

    Ggraph(const initializer_list<pair<int,int> >& list, const int n=-1): 
      obj(new Hgraph(n,list)){};

    Ggraph(const cnine::RtensorA& M):
      obj(new Hgraph(M)){}


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static Ggraph random(const int _n, const float p=0.5){
      return new Hgraph(Hgraph::random(_n,p));}

    static Ggraph edges(const cnine::RtensorA& M, int n=-1){
      if(n==-1) n=M.max()+1;
      return new Hgraph(M,n);}


  public: // ---- Access --------------------------------------------------------------------------------------


    int getn() const{
      return obj->getn();
    }

    cnine::RtensorA dense() const{
      return obj->dense();
    }

    bool operator==(const Ggraph& x) const{
      return obj==x.obj;
    }


  public: // ---- Operations ----------------------------------------------------------------------------------


    Ggraph permute(const cnine::permutation& pi) const{
      return Ggraph(new Hgraph(obj->permute(pi)));
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
