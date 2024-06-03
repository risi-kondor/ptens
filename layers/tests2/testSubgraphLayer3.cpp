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

  Subgraph triangle=Subgraph::cycle(3);
  cout<<triangle<<endl;

  Subgraph square=Subgraph::cycle(4);
  cout<<square<<endl;


  SubgraphLayer0<Ptensors0> f0(G,nc,cnine::fill_gaussian());
  cout<<f0<<endl;


  SubgraphLayer1<Ptensors1> f1(f0,triangle);
  cout<<"f1 done"<<endl;

  SubgraphLayer1<Ptensors1> f2(f0,square);
  cout<<"f2 done"<<endl;

  SubgraphLayer1<Ptensors1> f1d(f0,triangle);
  cout<<"f1d done"<<endl;


  Ptensors1 g1=Ptensors1::cat({f1,f2});
  cout<<"g1 done"<<endl;

  Ptensors1 g2=Ptensors1::cat({f1,f2});
  cout<<"g2 done"<<endl;


  SubgraphLayer1<Ptensors1> f3(g1,G,triangle);
  cout<<"f3 done"<<endl;

  SubgraphLayer1<Ptensors1> f4(g1,G,square);
  cout<<"f4 done"<<endl;


  Ptensors1 g3=Ptensors1::cat({f3,f4});
  cout<<"g3 done"<<endl;



  cout<<endl;
}
