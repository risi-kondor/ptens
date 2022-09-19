#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "LinMaps.hpp"
#include "ConcatFunctions.hpp"
#include "AddMsgFunctions.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  int N=8;
  Hgraph G=Hgraph::random(N,0.3);
  cout<<G.dense()<<endl;

  auto L0=Ptensors0::sequential(N,1);
  auto L1=concat(L0,G);
  PRINTL(L1);

  auto L2=Ptensors2::zero(G.nhoods(2),5);
  add_msg(L2,L1,G);
  PRINTL(L2);
  
  auto L3=Ptensors1::zero(G.nhoods(3),25);
  add_msg(L3,L2,G);
  PRINTL(L3);
  
}
