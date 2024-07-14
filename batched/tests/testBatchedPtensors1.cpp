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
#include "BatchedPtensors0.hpp"
#include "BatchedPtensors1.hpp"
#include "BatchedPtensors2.hpp"
#include "PtensSession.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  typedef BatchedPtensors0<float> BPtens0;
  typedef BatchedPtensors1<float> BPtens1;
  typedef BatchedPtensors2<float> BPtens2;

  AtomsPack xatoms=AtomsPack::random(4,4,0.5);
  BatchedAtomsPack bxatoms({xatoms,xatoms,xatoms});

  BPtens0 X0=BPtens0(bxatoms,channels=3,filltype=3);
  BPtens1 X1=BPtens1(bxatoms,channels=3,filltype=3);
  BPtens2 X2=BPtens2(bxatoms,channels=3,filltype=3);

  //cout<<X0<<endl;
  cout<<X1<<endl;
  //cout<<X2<<endl;

  auto Z1=BPtens1::linmaps(X1);
  cout<<Z1<<endl;
  BPtens1 X1ga=BPtens1::zeros_like(X1);
  X1ga.add_linmaps_back(Z1);
  cout<<X1ga<<endl;


  auto Z2=BPtens1::linmaps(X2);
  cout<<Z2<<endl;
  auto X2ga=BPtens2::zeros_like(X2);
  X2ga.add_linmaps_back(Z2);
  cout<<X2ga<<endl;


  AtomsPack yatoms=AtomsPack::random(4,4,0.5);
  BatchedAtomsPack byatoms({xatoms,xatoms,xatoms});

  auto Y0=BPtens1::gather(X0,byatoms);
  cout<<Y0<<endl;

  auto Y1=BPtens1::gather(X1,byatoms);
  cout<<Y1<<endl;

  auto Y2=BPtens1::gather(X2,byatoms);
  cout<<Y2<<endl;

  auto u=cat_channels(Y1,Y1);
  cout<<u<<endl;

  auto X0g=BPtens0::zeros_like(X0);
  auto X1g=BPtens1::zeros_like(X1);
  auto X2g=BPtens2::zeros_like(X2);

  X0g.add_gather_back(Y0);
  cout<<X0g<<endl;
  X1g.add_gather_back(Y1);
  cout<<X1g<<endl;
  X2g.add_gather_back(Y2);
  cout<<X2g<<endl;

  cout<<6666<<endl;
  auto C=BPtens1::cat({X1,X1,X1});
  cout<<C<<endl;

  //Ptensors1b<float> Ab(A);
  //cout<<Ab<<endl;

}

