#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  if(true){
    Ptensors2 A=Ptensors2::randn({{1,2,3},{3,5},{2}},2);
    cout<<A<<endl;
    auto Ag=A.to_device(1);
    cout<<Ag<<endl;
    auto B=A.to_device(0);
    cout<<B<<endl;
    exit(0);
  }

  Ptensors2 A=Ptensors2::sequential(2,3,3);
  cout<<A<<endl;
  cout<<linmaps0(A)<<endl;
  cout<<linmaps1(A)<<endl;
  cout<<linmaps2(A)<<endl;
  cout<<"-----"<<endl;

  #ifdef _WITH_CUDA
  Ptensors2 Ag(A,1);
  cout<<linmaps0(Ag)<<endl;
  cout<<linmaps1(Ag)<<endl;
  cout<<linmaps2(Ag)<<endl;
  #endif

}
