#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensors0.hpp"

#include "LinMaps.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensors0 A=Ptensors0::sequential(5,3);
  cout<<A<<endl;
  cout<<linmaps0(A)<<endl;
  //cout<<linmaps1(A)<<endl;

  Ptensors1 B=Ptensors1::sequential(5,3,3);
  cout<<B<<endl;
  cout<<linmaps0(B)<<endl;



}
