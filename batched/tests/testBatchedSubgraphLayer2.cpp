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
#include "BatchedSubgraphLayer2.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;

typedef BatchedPtensors0<float> BPtens0;
typedef BatchedPtensors1<float> BPtens1;


int main(int argc, char** argv){

  Ggraph g0=Ggraph::random(10);
  BatchedGgraph G({g0,g0,g0});
  cout<<G<<endl;

  Subgraph trivial=Subgraph::trivial();

  AtomsPack xatoms0=AtomsPack::random(10,10,0.5);
  BatchedAtomsPack xatoms({xatoms0,xatoms0,xatoms0});

  BPtens1 X1=BPtens1(xatoms,channels=3,filltype=3);

  BatchedSubgraphLayer2<float> U(X1,G,trivial);
  cout<<U<<endl;

}
