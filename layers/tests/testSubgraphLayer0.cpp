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
#include "SubgraphLayer0.hpp"
//#include "BatchedGgraph.hpp"

using namespace ptens;
using namespace cnine;



typedef Ptensors1<float> Ptens1;


int main(int argc, char** argv){

  PtensSession ptens_session;

  Ggraph G=Ggraph::random(10);
  //BatchedGgraph G({g0,g0,g0});
  cout<<G<<endl;

  Subgraph trivial=Subgraph::trivial();

  AtomsPack xatoms=AtomsPack::random(10,10,0.5);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=3);
  SubgraphLayer0<float> U(X1,G,trivial);
  cout<<U<<endl;

}
