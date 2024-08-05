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
#include "CompressedAtomsPack.hpp"
#include "PtensSession.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  AtomsPack _atoms=AtomsPack::random(6,6,0.5);
  CompressedAtomsPack atoms(_atoms,4,4);
  cout<<atoms<<endl;

}

