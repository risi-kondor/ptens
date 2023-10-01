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
#include "PtensSession.hpp"
#include "Ggraph.hpp"

#include "Subgraph.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens::ptens_session;


int main(int argc, char** argv){

  Ggraph M=Ggraph::random(10,0.2);
  cout<<M<<endl;

  Subgraph A=Subgraph::star(5);
  cout<<A.dense()<<endl;

}
