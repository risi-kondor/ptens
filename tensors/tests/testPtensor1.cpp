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
#include "LinmapFunctions.hpp"
#include "MsgFunctions.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  PtensSession session;

  auto A=Ptensor1<float>::sequential({0,1,2,3},1);
  cout<<A<<endl;

  cout<<linmaps0(A)<<endl;
  cout<<linmaps1(A)<<endl;
  cout<<linmaps2(A)<<endl;
  cout<<"---------------"<<endl<<endl;

  cout<<A<<endl;
  auto B=Ptensor1<float>::zero({0,1,5},2);
  A>>B; //add_msg(B,A);
  cout<<B<<endl;

  auto C=Ptensor2<float>::zero({0,1,5},5);
  A>>C; //add_msg(C,A);
  cout<<C<<endl;

  auto D=Ptensor1<float>::zero({2,3},2);
  cout<<(A>>D)<<endl;

}
