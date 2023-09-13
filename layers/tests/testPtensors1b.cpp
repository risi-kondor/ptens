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

#include "Ptensors1.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  Ptensors1 A0(3,2,2,cnine::fill_sequential());
  cout<<A0<<endl;

  Ptensors1 A1(2,3,2,cnine::fill_sequential());
  cout<<A1<<endl;

  Ptensors1 B=Ptensors1::cat({A0,A1});
  cout<<B<<endl;
}
