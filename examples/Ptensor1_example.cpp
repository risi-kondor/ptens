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
#include "Ptensor1.hpp"
#include "LinmapFunctions.hpp"
#include "MsgFunctions.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  PtensSession session;

  auto A=Ptensor1<float>::sequential({0,1,2,3},3);
  cout<<A<<endl;

  float v=A(3,1);
  cout<<v<<endl;

  auto M=Ltensor<float>::gaussian({3,2});
  Atoms D({5,8,3});
  Ptensor1 B(M,D);
  cout<<B<<endl;
  cout<<endl<<endl;

  auto P0=linmaps0(A);
  cout<<P0<<endl;

  auto P1=linmaps1(A);
  cout<<P1<<endl;

  auto P2=linmaps2(A);
  cout<<P2<<endl;

  cout<<"---------------"<<endl<<endl;

  auto M0=Ptensor0<float>::zero({0,1,5},3);
  A>>M0;
  cout<<M0<<endl;

  auto M1=Ptensor1<float>::zero({0,1,5},6);
  A>>M1; //add_msg(C,A);
  cout<<M1<<endl;

  auto M2=Ptensor2<float>::zero({0,1,5},15);
  A>>M2;
  cout<<M2<<endl;

}
