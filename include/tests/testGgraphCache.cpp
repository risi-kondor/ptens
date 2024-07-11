/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
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
#include "Ggraph.hpp"
#include "Subgraph.hpp"

using namespace ptens;
using namespace cnine;

PtensSession session;


int main(int argc, char** argv){

  Ggraph M=Ggraph::random(5,0.5);
  cout<<M<<endl;
  auto E=M.edge_list();

  //cout<<Ggraph::cached_from_edge_list(E)<<endl;
  //cout<<Ggraph::cached_from_edge_list(E)<<endl;
  //cout<<Ggraph::cached_from_edge_list(E)<<endl;

}
