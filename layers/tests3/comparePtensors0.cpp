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
#include "Ptensors0b.hpp"
#include "Ptensors1b.hpp"
#include "Ptensors2b.hpp"
#include "EMPlayers2.hpp"
#include "LinmapLayers.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  const int nc=3;
  typedef Ptensors0b<float> Ptens0;
  typedef Ptensors1b<float> Ptens1;
  typedef Ptensors2b<float> Ptens2;

  AtomsPack xatoms0=AtomsPack::random0(5);
  AtomsPack xatoms=AtomsPack::random(5,0.5);
  Ptensors0 x0=Ptensors0::gaussian(xatoms0,nc);
  Ptensors1 x1=Ptensors1::gaussian(xatoms,nc);
  Ptensors2 x2=Ptensors2::gaussian(xatoms,nc);

  Ptens0 X0(x0);
  Ptens1 X1(x1);
  Ptens2 X2(x2);

  AtomsPack yatoms=AtomsPack::random0(5);
  TransferMap tmap0=yatoms.overlaps(xatoms0);
  TransferMap tmap=yatoms.overlaps(xatoms);

  Ptensors0 y0=Ptensors0::zero(yatoms,nc);
  emp00(y0,x0,tmap0);
  Ptens0 Y0=Ptens0::gather(X0,yatoms);
  //cout<<y0<<endl;
  //cout<<Y0<<endl;
  cout<<"0 <- 0 error: "<<Y0.diff2(Ptens0(y0))<<endl;


  Ptensors0 y1=Ptensors0::zero(yatoms,nc);
  emp01(y1,x1,tmap);
  Ptens0 Y1=Ptens0::gather(X1,yatoms);
  //cout<<y1<<endl;
  //cout<<Y1<<endl;
  cout<<"0 <- 1 error: "<<Y1.diff2(Ptens0(y1))<<endl;

  Ptensors0 y2=Ptensors0::zero(yatoms,2*nc);
  emp02(y2,x2,tmap);
  Ptens0 Y2=Ptens0::gather(X2,yatoms);
  //cout<<y2<<endl;
  //cout<<Y2<<endl;
  cout<<"0 <- 2 error: "<<Y2.diff2(Ptens0(y2))<<endl;

  Ptensors0 z0=linmaps0(x0);
  Ptens0 Z0=ptens::linmaps0(X0);
  cout<<"linmaps(0 <- 0) error: "<<Z0.diff2(Ptens0(z0))<<endl;

  Ptensors0 z1=linmaps0(x1);
  Ptens0 Z1=ptens::linmaps0(X1);
  cout<<"linmaps(0 <- 1) error: "<<Z0.diff2(Ptens0(z0))<<endl;

  Ptensors0 z2=linmaps0(x2);
  Ptens0 Z2=ptens::linmaps0(X2);
  cout<<"linmaps(0 <- 2) error: "<<Z0.diff2(Ptens0(z0))<<endl;
  

}
