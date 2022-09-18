#include "Hgraph.hpp"
#include "CatFuncations.hpp"


using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  int N=8;
  Hgraph G=Hgraph::random(N,0.3);

  Ptensors A=Ptensors::sequential(N,1);
  cout<<A<<endl;

  auto nh1=G.nhoods(1);
  cout<<nh1<<endl;

  auto B=concat(A,G,nh1);


}
