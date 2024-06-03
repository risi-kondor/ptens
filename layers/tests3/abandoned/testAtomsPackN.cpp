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
#include "PtensSession.hpp"
#include "AtomsPackN.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  AtomsPackN x0(0,{{0},{2},{3}});
  AtomsPackN y0(0,{{0},{1},{2},{4}});
  cout<<x0<<endl;
  cout<<y0<<endl;

  MessageList overlaps=x0.overlaps_mlist(y0);
  cout<<overlaps<<endl;
  
  MessageMap overlaps_tmap=x0.overlaps_mmap(y0);
  cout<<overlaps_tmap<<endl;



  Ltensor<float> X1(Gdims(6,10));
  Ltensor<float> Y1(Gdims(7,5),filltype=4);

  AtomsPack1 x1({{0,1},{1,2,3},{5}});
  AtomsPack1 y1({{0},{1,2,3},{4,5},{6}});
  cout<<x1<<endl;
  cout<<y1<<endl;

  MessageList overlaps1=x1.overlaps_mlist(y1);
  cout<<overlaps1<<endl;
  
  MessageMap overlaps_tmap1=x1.overlaps_mmap(y1);
  cout<<overlaps_tmap1<<endl;
  cout<<overlaps_tmap1.inv()<<endl;
  overlaps_tmap1(X1,Y1);
  cout<<X1<<endl;


}
