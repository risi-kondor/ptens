#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Hgraph.hpp"
#include "FindPlantedSubgraphs.hpp"

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
