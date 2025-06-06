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
#include "BatchedAtomsPack.hpp"
#include "PtensSession.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  int n=5;

  AtomsPack a=AtomsPack::random(n,n,0.5);
  cout<<a<<endl;

  cout<<1<<endl;
  BatchedAtomsPack<1> A({a,a,a});
  cout<<2<<endl;
  //cout<<A<<endl;

}
