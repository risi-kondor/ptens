#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "LinMaps.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensors0 A=Ptensors0::sequential(5,3);
  cout<<A<<endl;
  cout<<linmaps2(A)<<endl;

  Ptensors1 B=Ptensors1::sequential(5,3,3);
  cout<<B<<endl;
  cout<<linmaps2(B)<<endl;


}
