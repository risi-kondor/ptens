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
#include "CnineSession.hpp"

#include "Ptensors0b.hpp"
#include "Ptensors1b.hpp"
#include "Ptensors2b.hpp"
#include "EMPlayers2.hpp"

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

  //cout<<x0<<endl;
  //cout<<X0<<endl;
  //cout<<X1<<endl;
  //cout<<X2<<endl;

  AtomsPack yatoms=AtomsPack::random(5,0.5);
  TransferMap tmap0=yatoms.overlaps(xatoms0);
  TransferMap tmap=yatoms.overlaps(xatoms);

  Ptensors1 y0=Ptensors1::zero(yatoms,nc);
  emp10(y0,x0,tmap0);
  Ptens1 Y0=Ptens1::gather(X0,yatoms);
  //cout<<y0<<endl;
  //cout<<Y0<<endl;
  cout<<"1 <- 0 error: "<<Y0.diff2(Ptens1(y0))<<endl;


  Ptensors1 y1=Ptensors1::zero(yatoms,2*nc);
  emp11(y1,x1,tmap);
  Ptens1 Y1=Ptens1::gather(X1,yatoms);
  //cout<<y1<<endl;
  //cout<<Y1<<endl;
  cout<<"1 <- 1 error: "<<Y1.diff2(Ptens1(y1))<<endl;

  Ptensors1 y2=Ptensors1::zero(yatoms,5*nc);
  //cout<<y2<<"------------"<<endl;
  emp21(y2,x2,tmap);
  Ptens1 Y2=Ptens1::gather(X2,yatoms);
  cout<<"1 <- 2 error: "<<Y2.diff2(Ptens1(y2))<<endl;

  Ltensor<float> M({2*nc,2*nc},4); 
  //cout<<Y1<<endl;
  //cout<<mprod(Y1,M)<<endl;

}
