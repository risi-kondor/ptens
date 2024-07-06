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
#include "CnineSession.hpp"

#include "TensorLevelMap.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  int N=6;

  auto A=AtomsPack::random(N,N,0.3);
  cout<<A<<endl;

  auto B=AtomsPack::random(N,N,0.3);
  cout<<B<<endl;

  auto M=TensorLevelMap::all_overlapping(A,B);
  cout<<M<<endl;

}
