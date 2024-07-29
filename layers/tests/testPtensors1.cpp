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
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "PtensSession.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  int N=4;

  typedef Ptensors0<float> Ptens0;
  typedef Ptensors1<float> Ptens1;
  typedef Ptensors2<float> Ptens2;

  AtomsPack xatoms=AtomsPack::random(N,N,0.5);
  Ptens0 X0=Ptens0(xatoms,channels=3,filltype=3);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=3);
  Ptens2 X2=Ptens2(xatoms,channels=3,filltype=3);

  auto Z1=Ptens2::linmaps(X1);
  cout<<Z1<<endl;
  Ptens1 X1ga=Ptens1::zeros_like(X1);
  X1ga.add_linmaps_back(Z1);
  cout<<X1ga<<endl;

  auto Z2=Ptens1::linmaps(X2);
  cout<<Z2<<endl;
  Ptens2 X2ga=Ptens2::zeros_like(X2);
  X2ga.add_linmaps_back(Z2);
  cout<<X2ga<<endl;

  AtomsPack yatoms=AtomsPack::random(N,N,0.5);
  auto lmap=LayerMap::overlaps_map(yatoms,xatoms);
  auto Y0=Ptens1::gather(yatoms,X0);
  auto Y1=Ptens1::gather(yatoms,X1);
  auto Y2=Ptens1::gather(yatoms,X2);
  cout<<Y0<<endl;
  cout<<Y1<<endl;
  cout<<Y2<<endl;

  auto u=cat_channels(Y1,Y1);
  cout<<u<<endl;

  exit(0);


  Ptens1& a=Y1.get_grad();
  Ptens1 b(a);

  Ptens0 X0g=Ptens0::zeros_like(X0);
  Ptens1 X1g=Ptens1::zeros_like(X1);
  Ptens2 X2g=Ptens2::zeros_like(X2);

  X0g.add_gather_back(Y0,lmap);
  cout<<X0g<<endl;
  X1g.add_gather_back(Y1,lmap);
  cout<<X1g<<endl;
  X2g.add_gather_back(Y2,lmap);
  cout<<X2g<<endl;

  exit(0);
  //Ptens1 C=Ptens1::cat({X1,X1,X1});
  //cout<<C<<endl;

  //Ptensors1<float> Ab(A);
  //cout<<Ab<<endl;

}

