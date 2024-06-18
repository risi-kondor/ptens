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

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  cnine_session session;

#ifdef _WITH_CUDA
  cudaDeviceReset(); // why do we need this?
#endif 

  int N=100;
  int nc=256;
  int niter=1;

  typedef Ptensors0b<float> Ptens0;
  typedef Ptensors1b<float> Ptens1;
  typedef Ptensors2b<float> Ptens2;

  AtomsPack xatoms=AtomsPack::random(N,0.5);
  AtomsPack yatoms=AtomsPack::random(N,0.5);

  Ptens0 X0=Ptens0(xatoms,channels=nc,filltype=3);
  Ptens1 X1=Ptens1(xatoms,channels=nc,filltype=3);
  Ptens2 X2=Ptens2(xatoms,channels=nc,filltype=3);
#ifdef _WITH_CUDA
  Ptens0 X0g(X0,1); 
  Ptens1 X1g(X1,1); 
  Ptens2 X2g(X2,1); 
#endif 


  Ptens1 Y0=Ptens1::gather(X0,yatoms); 
  timed_block("Gather 1<-0 (CPU)",[&](){
    for(int i=0; i<niter; i++) Y0.add_gather(X0);});

#ifdef _WITH_CUDA
  Ptens1 Y0g=Ptens1::gather(X0g,yatoms);
  timed_block("Gather 1<-0 (GPU)",[&](){
      for(int i=0; i<niter; i++) Y0g.add_gather(X0g);});
#endif 


  Ptens1 Y1=Ptens1::gather(X1,yatoms); 
  timed_block("Gather 1<-1 (CPU)",[&](){
      for(int i=0; i<niter; i++) Y1.add_gather(X1);});

#ifdef _WITH_CUDA
  Ptens1 Y1g=Ptens1::gather(X1g,yatoms);
  timed_block("Gather 1<-1 (GPU)",[&](){
      for(int i=0; i<niter; i++) Y1g.add_gather(X1g);});
#endif 


  Ptens1 Y2=Ptens1::gather(X2,yatoms); 
  timed_block("Gather 1<-2 (CPU)",[&](){
      for(int i=0; i<niter; i++) Y2.add_gather(X2);});

#ifdef _WITH_CUDA
  Ptens1 Y2g=Ptens1::gather(X2g,yatoms);
  timed_block("Gather 1<-2 (GPU)",[&](){
      for(int i=0; i<niter; i++) Y2g.add_gather(X2g);});
#endif 


}

