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
#include "LinmapLayers.hpp"
#include "ConcatLayers.hpp"
#include "EMPlayers.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  PtensSession session;

  int N=8;
  Hgraph G=Hgraph::random(N,0.3);
  cout<<G.dense()<<endl;

  auto L0=Ptensors0::sequential(N,1);
  auto L1=concat(L0,G);
  PRINTL(L1);

  auto L2=Ptensors1::zero(G.nhoods(2),2);
  add_msg(L2,L1,G);
  PRINTL(L2);
  
}
