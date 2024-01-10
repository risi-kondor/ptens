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

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  typedef Ptensors0b<float> Ptens0;
  typedef Ptensors1b<float> Ptens1;
  typedef Ptensors2b<float> Ptens2;

  AtomsPack xatoms=AtomsPack::random(4,0.5);
  Ptens0 X0=Ptens0(xatoms,channels=3,filltype=3);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=3);
  Ptens2 X2=Ptens2(xatoms,channels=3,filltype=3);

  AtomsPack yatoms=AtomsPack::random(4,0.5);
  cout<<X0<<endl;
  cout<<Ptens1::gather(X0,yatoms)<<endl;
  cout<<Ptens1::gather(X1,yatoms)<<endl;
  cout<<Ptens1::gather(X2,yatoms)<<endl;


  //exit(0);
  Ptens1 C=Ptens1::cat({X1,X1,X1});
  cout<<C<<endl;

  //Ptensors1b<float> Ab(A);
  //cout<<Ab<<endl;

}

