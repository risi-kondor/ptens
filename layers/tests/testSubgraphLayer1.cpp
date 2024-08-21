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

using namespace ptens;
using namespace cnine;

typedef Ptensors1<float> Ptens1;

PtensSession ptens_session;


int main(int argc, char** argv){

  cnine_session session;

  //cudaDeviceReset();

  Ggraph G=Ggraph::random(10);
  cout<<G<<endl;

  Subgraph trivial=Subgraph::trivial();
  cout<<trivial<<endl;
  Subgraph edge=Subgraph::edge();
  cout<<edge<<endl;
  Subgraph triangle=Subgraph::triangle();
  cout<<triangle<<endl;

  SubgraphLayer0<float> f0(G,5,4,0);
  cout<<f0<<endl;

  SubgraphLayer1<float> f1=gather1(f0,triangle);
  cout<<"-------"<<endl;
  cout<<f1.str()<<endl;

  exit(0);
  SubgraphLayer1<float> f2=gather1(f1,edge);
  //f2.get_grad()=f2;

  //f1.get_grad().add_gather_back(f2);
  //cout<<f1.get_grad()<<endl;

  /*
  Ltensor<float> W({5,5},filltype=4);
  Ltensor<float> b({5},filltype=4);

  auto f2=linear_sg(f1,W,b);
  cout<<f2.str()<<endl;

  AtomsPack xatoms=AtomsPack::random(10,0.5);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=3);
  SubgraphLayer1<float> U(X1,G,trivial);
  cout<<U<<endl;
  */

}
