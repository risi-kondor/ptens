#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  #ifdef _WITH_CUDA

  Ptensors2 A=Ptensors2::sequential({{1,2,3}},2);
  Ptensors2 Ag(A,1);
  cout<<A<<endl;

  cout<<linmaps0(A)<<endl;
  cout<<linmaps0(Ag)<<endl;

  cout<<linmaps1(A)<<endl;
  cout<<linmaps1(Ag)<<endl;

  cout<<linmaps2(A)<<endl;
  cout<<linmaps2(Ag)<<endl;

  cout<<"-----"<<endl;

  #endif

}
