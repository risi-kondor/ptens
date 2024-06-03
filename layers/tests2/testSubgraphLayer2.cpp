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
#include "SubgraphLayer1.hpp"
#include "Ptensors2.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  const int n=10;
  const int nc=5;

  Ggraph G=Ggraph::random(n);
  cout<<G<<endl;

  Subgraph S=Subgraph::cycle(5);
  cout<<S<<endl;

  auto W=Tensor<float>::gaussian({S.n_eblocks(),nc,nc});

  permutation pi=permutation::random(n);
  cout<<pi<<endl;
  
  Ggraph Gd=G.permute(pi);
  cout<<Gd<<endl;




  SubgraphLayer0<Ptensors0> f0(G,nc,cnine::fill_gaussian());
  auto f0d=f0.permute(pi);
  //cout<<f0<<endl;
  //cout<<f0d<<endl;
  //exit(0);

  SubgraphLayer1<Ptensors1> f1(f0,S);
  SubgraphLayer1<Ptensors1> f1d(f0d,S);
  //cout<<f1<<endl;

  //SubgraphLayer1<Ptensors1> f2=f1.autobahn(W);
  //SubgraphLayer1<Ptensors1> f2d=f1d.autobahn(W);
  //SubgraphLayer1<Ptensors1> f2dd=f2d.permute(pi.inv());
  //cout<<f2<<endl;
  //cout<<f2d<<endl;
  //cout<<f2dd<<endl;

}
