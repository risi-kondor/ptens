#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor2pack.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensor2pack layer0;

  layer0.push_back(Ptensor2::zero(Atoms::sequential(3),5));
  layer0.push_back(Ptensor2::sequential(Atoms::sequential(3),5));
  layer0.push_back(Ptensor2::zero(Atoms::sequential(4),5));
  cout<<layer0<<endl;

  Cgraph G;
  G.push(0,0);
  G.push(1,1);
  G.push(2,2);

  //layer0.forwardMP(layer0,G);
  //cout<<layer0<<endl;

  //Ptensor1pack layer1=layer0.fwd(G);
  //cout<<layer1<<endl;

}
