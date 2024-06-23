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
#include "AtomsPack.hpp"

#include "Cnine_base.cpp"

using namespace ptens;
using namespace cnine;

//PtensSession ptens::ptens_session;


int main(int argc, char** argv){

  //cnine_session session;

  AtomsPack x({{0,1},{1,2,3},{5}});
  AtomsPack y({{0},{1,2,3},{4,5},{6}});

  Hgraph G=Hgraph::overlaps(x,y);
  cout<<G.dense()<<endl;

}