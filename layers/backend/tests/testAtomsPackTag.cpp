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
#include "PtensSession.hpp"
#include "AtomsPackTag.hpp"

using namespace ptens;
using namespace cnine;

PtensSession session;


int main(int argc, char** argv){

  AtomsPack A({{1,2,3},{4,5}});
  AtomsPack B({{1,2,3},{4,5}});
  cout<<A<<endl;

  /*
  auto tag0=AtomsPackTag0::make(A.obj);
  auto tag1=AtomsPackTag0::make(A.obj);
  auto tag2=AtomsPackTag0::make(B.obj);

  //cout<<tag0->atoms()->str()<<endl;
  //cout<<(*tag0)->str()<<endl;
  cout<<(**tag0)<<endl;
  cout<<(**tag1)<<endl;

  cout<<&(**tag0)<<endl;
  cout<<&(**tag1)<<endl;
  cout<<&(**tag2)<<endl;
  */

  auto tag0=AtomsPackTag0(A.obj);
  auto tag1=AtomsPackTag0(A.obj);
  auto tag2=AtomsPackTag0(B.obj);

  cout<<*tag0<<endl;
  cout<<*tag1<<endl;
  cout<<*tag2<<endl;

  cout<<tag0.obj.get()<<endl;
  cout<<tag1.obj.get()<<endl;
  cout<<tag2.obj.get()<<endl;

}

