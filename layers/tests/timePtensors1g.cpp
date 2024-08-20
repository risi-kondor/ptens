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
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "PtensSession.hpp"

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

  typedef Ptensors0<float> Ptens0;
  typedef Ptensors1<float> Ptens1;
  typedef Ptensors2<float> Ptens2;

  AtomsPack xatoms=AtomsPack::random(N,N,0.5);
  AtomsPack yatoms=AtomsPack::random(N,N,0.5);
  LayerMap map=LayerMap::overlaps_map(xatoms,yatoms);

  Ptens0 X0=Ptens0(xatoms,channels=nc,filltype=3);
  Ptens1 X1=Ptens1(xatoms,channels=nc,filltype=3);
  Ptens2 X2=Ptens2(xatoms,channels=nc,filltype=3);
#ifdef _WITH_CUDA
  Ptens0 X0g(X0,1); 
  Ptens1 X1g(X1,1); 
  Ptens2 X2g(X2,1); 
#endif 


  Ptens1 Y0=Ptens1::gather(yatoms,X0); 
  TimedBlock("Gather 1<-0 (CPU)",[&](){
      for(int i=0; i<niter; i++) Y0.add_gather(X0,map);});

#ifdef _WITH_CUDA
  Ptens1 Y0g=Ptens1::gather(yatoms,X0g);
  TimedBlock("Gather 1<-0 (GPU)",[&](){
      for(int i=0; i<niter; i++) Y0g.add_gather(X0g,map);});
#endif 


  Ptens1 Y1=Ptens1::gather(yatoms,Y1); 
  TimedBlock("Gather 1<-1 (CPU)",[&](){
      for(int i=0; i<niter; i++) Y1.add_gather(X1,map);});

#ifdef _WITH_CUDA
  Ptens1 Y1g=Ptens1::gather(yatoms,Y1g);
  TimedBlock("Gather 1<-1 (GPU)",[&](){
      for(int i=0; i<niter; i++) Y1g.add_gather(X1g,map);});
#endif 


  Ptens1 Y2=Ptens1::gather(yatoms,X2); 
  TimedBlock("Gather 1<-2 (CPU)",[&](){
      for(int i=0; i<niter; i++) Y2.add_gather(X2,map);});

#ifdef _WITH_CUDA
  Ptens1 Y2g=Ptens1::gather(yatoms,X2g);
  TimedBlock("Gather 1<-2 (GPU)",[&](){
      for(int i=0; i<niter; i++) Y2g.add_gather(X2g,map);});
#endif 


}

