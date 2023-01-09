#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Hgraph.hpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  Hgraph M=Hgraph::random(10,0.2);

  M.set(2,3,5.0);
  M.set(2,8,1.0);
  M.set(3,1,4.0);

  cout<<M<<endl;
  
  CSRmatrix<float> Md=M.csrmatrix();
  cout<<Md<<endl;

  GatherMap Mg=M.broadcast_map();
  cout<<Mg<<endl;
  //for(int i=0; i<5; i++)
  //cout<<M.nhoods(i)<<endl;

  auto E=M.edges();
  cout<<E<<endl;

}

