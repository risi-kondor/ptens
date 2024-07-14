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

#include "Ptens_base.cpp"
#include "BatchedPtensors0.hpp"
#include "PtensSession.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  typedef BatchedPtensors0<float> Ptens0;
  //typedef BatchedPtensors1b<float> Ptens1;
  //typedef BatchedPtensors2b<float> Ptens22;

  AtomsPack xatoms=AtomsPack::random(8,8,0.5);
  Ptensors0<float> x0=Ptensors0<float>(xatoms,channels=3,filltype=4);

  Ptens0 X0({x0,x0,x0});
  cout<<X0<<endl;

  Ptens0 Y0=Ptens0::linmaps(X0);
  cout<<Y0<<endl;

  //Ltensor<float> M({8,2},filltype=3);
  //Ptens0 Z(M,{2,3,3});
  //cout<<Z<<endl;

}


