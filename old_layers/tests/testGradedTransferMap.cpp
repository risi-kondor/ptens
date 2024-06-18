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
#include "SubgraphLayer0.hpp"
#include "SubgraphLayer1.hpp"
#include "Ptensors2.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  int N=8;
  int nc=1;

  Hgraph G=Hgraph::random(N,0.3);
  cout<<G.dense()<<endl;

  auto L0=Ptensors1::sequential(G.nhoods(1),nc);
  PRINTL(L0);

  auto L1=Ptensors1::zero(G.nhoods(2),2*nc);
  auto L2=Ptensors1::zero(G.nhoods(2),2*nc);
  auto L3=Ptensors1::zero(G.nhoods(2),2*nc);

  Hgraph T0=Hgraph::overlaps(L1.atoms,L0.atoms);
  add_msg(L1,L0,T0);
  PRINTL(L1);

  TransferMap T1(new TransferMapObj<AtomsPackObj>(*L0.atoms.obj,*L1.atoms.obj));
  emp11(L2,L0,T1);
  PRINTL(L2);

  TransferMap T2(new TransferMapObj<AtomsPackObj>(*L0.atoms.obj,*L1.atoms.obj,true));
  emp11(L3,L0,T2);
  PRINTL(L3);


  cout<<(L1==L2)<<endl;
  cout<<(L1==L3)<<endl;

}

