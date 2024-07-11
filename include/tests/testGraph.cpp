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
#include "Ggraph.hpp"
#include "Subgraph.hpp"

using namespace ptens;
using namespace cnine;

PtensSession ptens_session;


int main(int argc, char** argv){

  Tensor<float> A({{0.0, 1, 1, 1, 0, 0},
        {1.0, 0, 0, 1, 1, 1},
	  {1.0, 0, 0, 0, 1, 1},
	    {1.0, 1, 0, 0, 0, 1},
	      {0.0, 1, 1, 0, 0, 1},
		{0.0, 1, 1, 1, 1, 0}});
  cout<<A<<endl;

  Ggraph M=Ggraph(A);
  auto Z=Subgraph::cycle(5);
  cout<<Z<<endl;

  auto V=M.subgraphs(Z);
  cout<<V<<endl;

  //Ggraph M=Ggraph::random(5,0.5);
  cout<<M<<endl;

  Subgraph S=Subgraph::triangle();
  cout<<S<<endl;

  auto U=M.subgraphs(S);
  cout<<U<<endl;


  Tensor<int> B({{0,1,2},{1,2,0}});
  Tensor<int> D(Gdims(3));
  D.set(0,2);
  D.set(1,3);
  D.set(2,2);
  Subgraph P=Subgraph::edge_index_degrees(B,D);
  cout<<P<<endl;

  cout<<"-- Cache: --"<<endl;
  //cout<<Subgraph::cached()<<endl;
  cout<<ptens_global::subgraph_cache<<endl;

}
