#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Hgraph.hpp"
#include "FindPlantedSubgraphs.jpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  Hgraph M=Hgraph::random(5,0.5);
  cout<<M.dense();

  cout<<M<<endl;
  cout<<M.spanning_tree()<<endl;

}

