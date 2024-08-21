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

  //cudaSetDevice(0);
  //cudaDeviceSynchronize();
  cudaDeviceReset(); // why do we need this?

  typedef Ptensors0<float> Ptens0;
  typedef Ptensors1<float> Ptens1;
  typedef Ptensors2<float> Ptens2;

  AtomsPack xatoms=AtomsPack::random(4,4,0.5);
  Ptens0 X0=Ptens0(xatoms,channels=3,filltype=3);
  Ptens1 X1=Ptens1(xatoms,channels=3,filltype=3);
  Ptens2 X2=Ptens2(xatoms,channels=3,filltype=3);

  Ptens0 X0g(X0,1); 
  Ptens1 X1g(X1,1); 
  Ptens2 X2g(X2,1); 

  AtomsPack yatoms=AtomsPack::random(4,4,0.5);
  Ptens1 Y0=Ptens1::gather(yatoms,X0);
  Ptens1 Y1=Ptens1::gather(yatoms,X1);
  Ptens1 Y2=Ptens1::gather(yatoms,X2);

  Ptens1 Y0g=Ptens1::gather(yatoms,X0g);
  Ptens1 Y1g=Ptens1::gather(yatoms,X1g);
  Ptens1 Y2g=Ptens1::gather(yatoms,X1g);

  //cout<<Y0<<endl;
  //cout<<Y0g<<endl;
  //cout<<Y1<<endl;
  //cout<<Y1g<<endl;
  //cout<<Y2<<endl;
  //cout<<Y2g<<endl;

  cout<<Y0.diff2(Y0g)<<endl;
  cout<<Y1.diff2(Y1g)<<endl;
  cout<<Y2.diff2(Y2g)<<endl;

  //cudaDeviceSynchronize();
  //cudaDeviceReset();

  #endif 

}

