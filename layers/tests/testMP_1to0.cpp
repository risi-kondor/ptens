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
#include "CnineSession.hpp"

#include "LinmapLayers.hpp"
//#include "ConcatLayers.hpp"
#include "EMPlayers.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  cnine_session session;

  int N=8;
  Hgraph G=Hgraph::random(N,0.3);
  cout<<G.dense()<<endl;

  Ptensors0 L0=Ptensors0::sequential(N,1);
  Ptensors1 L1=concat(L0,G);
  PRINTL(L1);

  Ptensors0 L2=Ptensors0::zero(G.nhoods(0),1);
  add_msg(L2,L1,G);
  PRINTL(L2);
  
}
