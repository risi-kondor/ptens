
#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor2.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Atoms a1({1,2,3});
  Atoms a2({2,3,4});

  Ptensor1 A=Ptensor1::sequential(a1,1);
  cout<<Ptensor2(A,a2)<<endl;

  Ptensor2 B=Ptensor2::sequential(a1,1);
  cout<<Ptensor2(B,a2)<<endl;

  cout<<B.msg1(a2)<<endl;

}
