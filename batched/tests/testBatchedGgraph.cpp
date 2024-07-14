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
#include "BatchedGgraph.hpp"
#include "Subgraph.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  Ggraph M=Ggraph::random(5,0.5);
  cout<<M<<endl;

  Subgraph A=Subgraph::triangle();
  cout<<A<<endl;

  auto U=BatchedGgraph({M,M,M});
  cout<<U<<endl;

  Ggraph A1=Ggraph::random(4,0.5);
  Ggraph A2=Ggraph::random(6,0.5);
  A1.cache(77);
  A2.cache(78);

  BatchedGgraph C(vector<int>({77,77,78}));
  cout<<C<<endl;
}
