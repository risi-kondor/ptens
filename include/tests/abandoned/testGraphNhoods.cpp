#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "GraphNhoods.hpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  Cgraph G;
  G.push(0,1);
  G.push(1,0);
  G.push(1,2);
  G.push(2,1);
  G.push(2,3);
  G.push(3,2);

  GraphNhoods nhoods(G,3);

  cout<<nhoods.level(0)<<endl;
  cout<<nhoods.level(1)<<endl;
  cout<<nhoods.level(2)<<endl;

}

