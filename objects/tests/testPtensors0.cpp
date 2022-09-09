#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensors0.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensors0 A=Ptensors0::sequential(5,3);
  cout<<A<<endl;

  Ptensors0 B=A.hom();
  cout<<B<<endl;


}
