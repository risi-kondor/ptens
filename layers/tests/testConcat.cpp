#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Hgraph.hpp"

#include "ConcatLayers.hpp"


using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  int N=8;
  Hgraph G=Hgraph::random(N,0.3);
  cout<<G.dense()<<endl;

  Ptensors0 A=Ptensors0::sequential(N,1);
  cout<<A<<endl;

  auto B=concat(A,G);
  cout<<B<<endl;

}
