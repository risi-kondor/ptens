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
#ifndef _PtensorSubpackSpecializer
#define _PtensorSubpackSpecializer

#include "Cgraph.hpp"
#include "iipair.hpp"


namespace ptens{

  class PtensorSubgraphSpecializer{
  public:

    unordered_map<iipair,Cgraph*> graphs;

    ~PtensorSubgraphSpecializer(){
      for(auto p:graphs)
	delete p.second;
    }

    Cgraph& graph(const int i, const int j){
      auto it=graphs.find(iipair(i,j));
      if(it==graphs.end()){
	Cgraph* G=new Cgraph();
	graphs[iipair(i,j)]=G;
	return *G;
      }
      return *it->second;
    }

    void forall(const std::function<void(const int, const int, Cgraph&)>& lambda){
      for(auto p: graphs)
	lambda(p.first.i0,p.first.i1,*p.second);
    }

  };

}

#endif 
