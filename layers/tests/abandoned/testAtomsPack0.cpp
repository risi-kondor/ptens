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
#include "AtomsPack0.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  AtomsPack0 x({{0},{2},{3}});
  AtomsPack0 y({{0},{1},{2},{4}});
  cout<<x<<endl;
  cout<<y<<endl;

  MessageList overlaps=x.overlaps_mlist(y);
  cout<<overlaps<<endl;
  
  MessageMap overlaps_tmap=x.overlaps_mmap(y);
  cout<<overlaps_tmap<<endl;


}
