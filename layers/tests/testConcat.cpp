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
#include "Hgraph.hpp"

//#include "ConcatLayers.hpp"


using namespace ptens;
using namespace cnine;

//PtensSession ptens_session;


int main(int argc, char** argv){

  PtensSession session;

  int N=8;
  Hgraph G=Hgraph::random(N,0.3);
  cout<<G.dense()<<endl;

  //Ptensors0 A=Ptensors0::sequential(N,1);
  //cout<<A<<endl;

  //auto B=concat(A,G);
  //cout<<B<<endl;

}
