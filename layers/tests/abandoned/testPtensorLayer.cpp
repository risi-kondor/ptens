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
#include "EMPlayers.hpp"
#include "PtensorLayer.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  AtomsPack xatoms=AtomsPack::random(8,0.5);
  AtomsPack yatoms=AtomsPack::random(8,0.5);

  PtensorLayer<float> X0(0,xatoms,channels=3,filltype=3);
  PtensorLayer<float> X1(1,xatoms,channels=3,filltype=3);
  PtensorLayer<float> X2(2,xatoms,channels=3,filltype=3);

  //cout<<X0<<endl;
  //cout<<X1<<endl;
  //cout<<X2<<endl;

  cout<<PtensorLayer<float>::gather(0,yatoms,X0)<<endl;
  cout<<PtensorLayer<float>::gather(0,yatoms,X1)<<endl;
  cout<<PtensorLayer<float>::gather(0,yatoms,X2)<<endl;




}
