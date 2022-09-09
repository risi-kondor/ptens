#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensors1.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensors1 A=Ptensors1::sequential(5,3,3);
  cout<<A<<endl;

  Ptensors1 B=A.hom();
  cout<<B<<endl;


}
