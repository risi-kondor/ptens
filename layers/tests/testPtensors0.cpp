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

typedef Ptensors0<float> Ptens0;
typedef Ptensors1<float> Ptens1;
typedef Ptensors2<float> Ptens2;

PtensSession session;


int main(int argc, char** argv){

  int N=6;

  //session.row_level_operations(true);
  //session.cache_overlap_maps(true);
  //session.cache_rmaps(true);
  ///cout<<session<<endl;
  

  AtomsPack xatoms=AtomsPack::random(N,N,0.5);
  cout<<xatoms<<endl;

  Ptens0 X0=Ptens0(xatoms,channels=3,filltype=4);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=4);
  Ptens2 X2=Ptens2(xatoms,channels=3,filltype=4);
  cout<<X0<<endl;
  //cout<<X1<<endl;
  //cout<<X2<<endl;

  
  //Ptens0 XX=Ptens0::cat({X0,X0});
  //cout<<"Cat:"<<endl;
  //cout<<XX<<endl;

  auto Z0=Ptens0::linmaps(X0);
  auto Z1=Ptens0::linmaps(X1);
  auto Z2=Ptens0::linmaps(X2);
  cout<<"Linmaps:"<<endl<<endl;
  cout<<Z0<<endl;
  cout<<Z1<<endl;
  cout<<Z2<<endl;


  Ptens0 X0ga=Ptens0::zeros_like(X0);
  Ptens1 X1ga=Ptens1::zeros_like(X1);
  Ptens2 X2ga=Ptens2::zeros_like(X2);
  X0ga.add_linmaps_back(Z0);
  X1ga.add_linmaps_back(Z1);
  X2ga.add_linmaps_back(Z2);
  cout<<"Linmaps_back:"<<endl<<endl;
  cout<<X0ga<<endl;
  cout<<X1ga<<endl;
  cout<<X2ga<<endl;

  AtomsPack yatoms=AtomsPack::random(N,N,0.5);
  cout<<"Gather:"<<endl;
  cout<<Ptens0::gather(yatoms,X0)<<endl;
  cout<<Ptens0::gather(yatoms,X1)<<endl;
  cout<<Ptens0::gather(yatoms,X2)<<endl;

  cout<<session<<endl;



}
