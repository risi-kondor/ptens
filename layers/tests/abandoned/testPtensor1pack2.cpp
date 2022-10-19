#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor1pack.hpp"
#include "GraphNhoods.hpp"
#include "ConcatenatingLayer0to1.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  const int n=6;

  Cgraph G=Cgraph::random(n,0.4);
  cout<<G.tensor()<<endl;

  /*
  GraphNhoods nhoods(G,3);
  Ptensor0pack layer0=Ptensor0pack::sequential(nhoods.level(0),5);
  Ptensor1pack layer1=Ptensor1pack::zero(nhoods.level(1),5);
  Ptensor1pack layer2=Ptensor1pack::zero(nhoods.level(2),5);

  ConcatenatingLayer0to1().forward(layer1,layer0);

  cout<<layer0<<endl;
  cout<<layer1<<endl;
  */


  /*
  Ptensor1pack layer0;

  layer0.push_back(Ptensor1::zero(Atoms::sequential(3),5));
  layer0.push_back(Ptensor1::sequential(Atoms::sequential(3),5));
  layer0.push_back(Ptensor1::zero(Atoms::sequential(4),5));
  cout<<layer0<<endl;

  Cgraph G;
  G.push(0,0);
  G.push(1,1);
  G.push(2,2);

  //layer0.forwardMP(layer0,G);
  //cout<<layer0<<endl;

  Ptensor1pack layer1=layer0.fwd(G);
  cout<<layer1<<endl;
  */

}
