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

#include "Cnine_base.cpp"
#include "Ptens_base.cpp"
#include "BatchedPtensors0.hpp"
#include "BatchedPtensors1.hpp"
#include "BatchedSubgraphLayer1.hpp"
//#include "BatchedSubgraphLayer1.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;

typedef BatchedPtensors0<float> BPtens0;
typedef BatchedPtensors1<float> BPtens1;


int main(int argc, char** argv){
  
  int n=6; 

  Ggraph g0=Ggraph::random(n);
  BatchedGgraph G({g0,g0,g0});
  cout<<G<<endl;

  Subgraph trivial=Subgraph::trivial();

  AtomsPack xatoms0=AtomsPack::random(n,n,0.5);
  BatchedAtomsPack xatoms({xatoms0,xatoms0,xatoms0});

  BPtens1 X1=BPtens1(xatoms,channels=3,filltype=3);

  BatchedSubgraphLayer1<float> U(X1,G,trivial);
  cout<<U<<endl;

  cout<<66666<<endl;
  cout<<BatchedSubgraphLayer1<float>::linmaps(U)<<endl;
  cout<<77777<<endl;

  auto edges=g0.edge_list();
  cout<<edges<<endl;

  Ggraph g1=Ggraph::from_edges(edges);
  g1.cache(32);
  //cout<<g1.original_edges()<<endl;
  cout<<Ggraph(32)<<endl;
  auto g2=Ggraph(32);
  cout<<g2.getn()<<endl;
  cout<<g1.nedges()<<endl;

  //Ltensor<float> M({12*g1.nedges(),3},3);
  //auto V=BatchedSubgraphLayer1<float>::from_edge_features({32,32,32},M);
  //cout<<V<<endl;

}
