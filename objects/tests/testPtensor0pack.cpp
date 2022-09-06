#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor0pack.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensor0pack layer0;

  layer0.push_back(Ptensor0::zero(Atoms::sequential(3),5));
  layer0.push_back(Ptensor0::sequential(Atoms::sequential(3),5));
  layer0.push_back(Ptensor0::zero(Atoms::sequential(4),5));
  cout<<layer0<<endl;

  Cgraph G;
  G.push(0,0);
  G.push(1,1);
  G.push(2,2);


  //Ptensor1pack layer1=layer0.fwd(G);
  //cout<<layer1<<endl;

}
