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



int main(int argc, char** argv){

  PtensSession session;
  int N=4;

  typedef Ptensors0<float> Ptens0;
  typedef Ptensors1<float> Ptens1;
  typedef Ptensors2<float> Ptens2;

  AtomsPack xatoms=AtomsPack::random(N,N,0.5);
  AtomsPack yatoms=AtomsPack::random(N,N,0.5);

  Ptens0 X0=Ptens0(xatoms,channels=3,filltype=3);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=3);

  auto Y00=Ptens0::gather(yatoms,X0);
  auto Y01=Ptens0::gather(yatoms,X1);
  auto Y10=Ptens1::gather(yatoms,X0);
  auto Y11=Ptens1::gather(yatoms,X1);
  cout<<Y00<<endl;
  cout<<Y01<<endl;
  cout<<Y10<<endl;
  cout<<Y11<<endl;

  //cout<<"-------------------------------"<<endl;
  //session.row_level_operations(true);

  /*
  auto Y00r=Ptens0::gather(X0,yatoms);
  auto Y01r=Ptens0::gather(X1,yatoms);
  auto Y10r=Ptens1::gather(X0,yatoms);
  auto Y11r=Ptens1::gather(X1,yatoms);
  cout<<Y00r<<endl;
  cout<<Y01r<<endl;
  cout<<Y10r<<endl;
  cout<<Y11r<<endl;
  */
  


}

