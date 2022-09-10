#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "LinMaps.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  auto A=Ptensor2::gaussian({0,1,2,3},3);
  cout<<A<<endl;
  cout<<linmaps0(A)<<endl;
  cout<<linmaps1(A)<<endl;
  cout<<linmaps2(A)<<endl;



}

  /*
  cnine_session session;

  Atoms a1({1,2,3});
  Atoms a2({2,3,4});

  Ptensor1 A=Ptensor1::sequential(a1,1);
  cout<<Ptensor2(A,a2)<<endl;

  Ptensor2 B=Ptensor2::sequential(a1,1);
  cout<<Ptensor2(B,a2)<<endl;

  cout<<B.msg1(a2)<<endl;
  */
