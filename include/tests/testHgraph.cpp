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
#include "Hgraph.hpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  Hgraph M=Hgraph::random(10,0.2);

  M.set(2,3,5.0);
  M.set(2,8,1.0);
  M.set(3,1,4.0);

  cout<<M<<endl;
  
  CSRmatrix<float> Md=M.csrmatrix();
  cout<<Md<<endl;

  GatherMap Mg=M.broadcast_map();
  cout<<Mg<<endl;
  //for(int i=0; i<5; i++)
  //cout<<M.nhoods(i)<<endl;

  auto E=M.edges();
  cout<<E<<endl;

}

