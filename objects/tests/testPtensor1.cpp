
#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor1.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Atoms a1({1,2,3});
  Atoms a2({2,3,4});

  cout<<a1.intersect(a2)<<endl;

  Ptensor1 A=Ptensor1::sequential(a1,1);
  cout<<A<<endl;

  //Ptensor1 B=Ptensor1::zero(a2,1);
  //B.gather(A);
  Ptensor1 B(A,a2);
  cout<<B<<endl;

}
