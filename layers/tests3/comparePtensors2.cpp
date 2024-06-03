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

  const int nc=1;
  typedef Ptensors0b<float> Ptens0;
  typedef Ptensors1b<float> Ptens1;
  typedef Ptensors2b<float> Ptens2;

  AtomsPack xatoms0=AtomsPack::random0(5);
  AtomsPack xatoms=AtomsPack::random(5,0.5);
  Ptensors0 x0=Ptensors0::sequential(xatoms0,nc);
  Ptensors1 x1=Ptensors1::sequential(xatoms,nc);
  Ptensors2 x2=Ptensors2::sequential(xatoms,nc);

  Ptens0 X0(x0);
  Ptens1 X1(x1);
  Ptens2 X2(x2);

  AtomsPack yatoms=AtomsPack::random(5,0.5);
  TransferMap tmap0=yatoms.overlaps(xatoms0);
  TransferMap tmap=yatoms.overlaps(xatoms);
  //cout<<yatoms<<endl;

  Ptensors2 y0=Ptensors2::zero(yatoms,2*nc); emp20(y0,x0,tmap0);
  Ptens2 Y0=Ptens2::gather(X0,yatoms);
  cout<<"2 <- 0 error: "<<Y0.diff2(Ptens2(y0))<<endl;

  Ptensors2 y1=Ptensors2::zero(yatoms,5*nc); emp21(y1,x1,tmap);
  Ptens2 Y1=Ptens2::gather(X1,yatoms);
  cout<<"2 <- 1 error: "<<Y1.diff2(Ptens2(y1))<<endl;
  //cout<<y1(0)<<endl;
  //cout<<Y1(0)<<endl;

  Ptensors2 y2=Ptensors2::zero(yatoms,15*nc); emp22(y2,x2,tmap);
  Ptens2 Y2=Ptens2::gather(X2,yatoms);
  cout<<"2 <- 2 error: "<<Y2.diff2(Ptens2(y2))<<endl;
  //cout<<y2(0)<<endl;
  //cout<<Y2(0)<<endl;

  Ptensors2 z0=linmaps2(x0);
  Ptens2 Z0=ptens::linmaps2(X0);
  cout<<"linmaps(2 <- 0) error: "<<Z0.diff2(Ptens2(z0))<<endl;

  Ptensors2 z1=linmaps2(x1);
  Ptens2 Z1=ptens::linmaps2(X1);
  cout<<"linmaps(2 <- 1) error: "<<Z0.diff2(Ptens2(z0))<<endl;

  Ptensors2 z2=linmaps2(x2);
  Ptens2 Z2=ptens::linmaps2(X2);
  cout<<"linmaps(2 <- 2) error: "<<Z0.diff2(Ptens2(z0))<<endl;
  cout<<z2<<endl;
  cout<<Z2<<endl;

}
