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
#include "SubgraphLayer1b.hpp"
#include "Ptensors2b.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  Ggraph G=Ggraph::random(10);
  cout<<G<<endl;

  Subgraph S=Subgraph::triangle();
  cout<<S<<endl;

  SubgraphLayer1b<float> f0(G,S,3,4,0);
  cout<<f0<<endl;

  Ltensor<float> W({2,3,4},4);
  Ltensor<float> B({2,4},4);

  Ltensor<float> Wg({2,3,4},0);
  Ltensor<float> Bg({2,4},0);

  SubgraphLayer1b f1=f0.autobahn(W,B);
  cout<<f1<<endl;
  f1.add_to_grad(f1);

  f0.add_autobahn_back0(f1,W);
  cout<<f0.get_grad()<<endl;

  f0.add_autobahn_back1_to(Wg,Bg,f1);
  cout<<Wg<<endl;
  cout<<Bg<<endl;
}
