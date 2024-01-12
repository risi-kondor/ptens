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

#include "SubgraphLayer0b.hpp"
#include "SubgraphLayer1b.hpp"
#include "SubgraphLayer2b.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens::ptens_session;

typedef Ptensors1b<float> Ptens1;


int main(int argc, char** argv){

  Ggraph G=Ggraph::random(10);
  cout<<G<<endl;

  Subgraph trivial=Subgraph::trivial();
  cout<<trivial<<endl;
  Subgraph edge=Subgraph::edge();
  cout<<edge<<endl;
  Subgraph triangle=Subgraph::triangle();
  cout<<triangle<<endl;

  SubgraphLayer0b<float> f0(G,5,4);
  cout<<f0<<endl;

  SubgraphLayer2b<float> f2(f0,edge);
  cout<<f2<<endl;

}
