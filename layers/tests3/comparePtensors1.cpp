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
  Ptensors0 x0=Ptensors0::sequential(xatoms0,nc);
  Ptensors1 x1=Ptensors1::sequential(xatoms,nc);
  Ptensors2 x2=Ptensors2::sequential(xatoms,nc);

  Ptens0 X0(x0);
  Ptens1 X1(x1);
  Ptens2 X2(x2);

  cout<<x0<<endl;
  cout<<X0<<endl;
  //cout<<X1<<endl;
  //cout<<X2<<endl;

  AtomsPack yatoms=AtomsPack::random(5,0.5);
  TransferMap tmap=yatoms.overlaps(xatoms);
  cout<<*tmap.obj<<endl;

  Ptensors1 y0=Ptensors1::zero(yatoms,nc);
  Ptensors1 y1=Ptensors1::zero(yatoms,2*nc);
  Ptensors1 y2=Ptensors1::zero(yatoms,5*nc);

  emp10(y0,x0,tmap);
  cout<<y0<<endl;
  cout<<Ptens1::gather(x0,yatoms)<<endl;
  exit(0);

  emp11(y1,x1,tmap);
  cout<<y1<<endl;
  cout<<Ptens1::gather(x1,yatoms)<<endl;

  emp12(y2,x2,tmap);
  cout<<y2<<endl;
  cout<<Ptens1::gather(x2,yatoms)<<endl;

  

}
