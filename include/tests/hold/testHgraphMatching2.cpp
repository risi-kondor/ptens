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
#include "PtensFindPlantedSubgraphs.hpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  RtensorA L=RtensorA::sequential({3});

  Hgraph triangle(3,{{0,1},{1,2},{2,0}},L);
  Hgraph square(4,{{0,1},{1,2},{2,3},{3,0}});

  cout<<triangle.str()<<endl;

  //Hgraph G=Hgraph::random(5,0.5);
  Hgraph G(8,RtensorA::sequential(8));
  G.insert(triangle,{0,1,2});
  G.insert(triangle,{5,6,7});
  cout<<G.dense();

  //cout<<G<<endl;
  //cout<<G.greedy_spanning_tree()<<endl;

  auto fn=FindPlantedSubgraphs(G,triangle);
  AindexPack sets(fn);
  cout<<sets<<endl;

  cout<<CachedPlantedSubgraphs()(G,triangle)<<endl;
  cout<<CachedPlantedSubgraphs()(G,triangle)<<endl;

}

