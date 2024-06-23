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
#include "Ptensors1b.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  AtomsPack atoms({{1,3,4},{5,3},{0,2}});

  auto A=Ptensors1b<float>::gaussian(atoms,3);
  cout<<A<<endl;

  auto B1=Ptensors1b<float>::linmaps(A);
  cout<<B1<<endl;

}
