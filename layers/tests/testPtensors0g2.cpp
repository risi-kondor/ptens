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
#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"
#include "GatherLayers.hpp"


using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


template<typename TYPE>
Ptensors0 backward_linmap(const Ptensors0& x, const TYPE& G){
  Ptensors0 R=Ptensors0::zeros_like(x);
  add_linmaps_back(R,G);
  return R;
}

template<typename TYPE>
Ptensors0 backward_unite(const Ptensors0& x, const TYPE& g, const Hgraph& G){
  Ptensors0 r=Ptensors0::zeros_like(x);
  add_msg_back(r,g,G);
  return r;
}


int main(int argc, char** argv){

  #ifdef _WITH_CUDA

  Ptensors0 A=Ptensors0::randn(3,3);
  Ptensors0 Ag(A,1);
  //cout<<A<<endl;

  Hgraph G=Hgraph::randomd(3,0.3);
  cout<<G<<endl;

  {
    auto B=linmaps0(A);
    cout<<"linmaps0:"<<B.diff2(linmaps0(Ag))<<endl;
    Ptensors0 G=Ptensors0::gaussian_like(B);
    auto Aback=backward_linmap(A,G);
    auto Abackg=backward_linmap(Ag,G.to_device(1));
    cout<<Aback.diff2(Abackg)<<endl;
  }

 {
    auto B=linmaps1(A);
    cout<<"linmaps1:"<<B.diff2(linmaps1(Ag))<<endl;
    Ptensors1 G=Ptensors1::gaussian_like(B);
    auto Aback=backward_linmap(A,G);
    auto Abackg=backward_linmap(Ag,G.to_device(1));
    cout<<Aback.diff2(Abackg)<<endl;
  }

 {
    auto B=linmaps2(A);
    cout<<"linmaps2:"<<B.diff2(linmaps2(Ag))<<endl;
    Ptensors2 G=Ptensors2::randn_like(B);
    auto Aback=backward_linmap(A,G);
    auto Abackg=backward_linmap(Ag,G.to_device(1));
    cout<<Aback.diff2(Abackg)<<endl;
  }

  {
    auto B=unite1(A,G);
    cout<<"unite1:"<<B.diff2(unite1(Ag,G))<<endl;
    Ptensors1 g=Ptensors1::randn_like(B);
    auto Aback=backward_unite(A,g,G);
    auto Abackg=backward_unite(Ag,g.to_device(1),G);
    cout<<Aback.diff2(Abackg)<<endl;
  }

  {
    auto B=unite2(A,G);
    cout<<"unite2:"<<B.diff2(unite2(Ag,G))<<endl;
    Ptensors2 g=Ptensors2::randn_like(B);
    auto Aback=backward_unite(A,g,G);
    auto Abackg=backward_unite(Ag,g.to_device(1),G);
    cout<<Aback.diff2(Abackg)<<endl;
  }

  #endif


}
