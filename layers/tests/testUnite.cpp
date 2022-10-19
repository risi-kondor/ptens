#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "EMPlayers.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  const int n=8;
  Hgraph G=Hgraph::random(n,0.3);
  cout<<G<<endl;

  Ptensors0 x=Ptensors0::sequential(n,2);
  cout<<x<<endl;

  cout<<unite1(x,G)<<endl;
  

}
