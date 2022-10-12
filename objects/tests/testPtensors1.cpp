#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "LinmapFunctions.hpp"
#include "MsgFunctions.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  if(true){
    Ptensors1 A=Ptensors1::randn({{1,2,3},{3,5},{2}},2);
    cout<<A<<endl;
    auto Ag=A.to_device(1);
    cout<<Ag<<endl;
    auto B=A.to_device(0);
    cout<<B<<endl;
    exit(0);
  }

  Ptensors1 A=Ptensors1::sequential(2,3,1);
  cout<<A<<endl;
  cout<<linmaps0(A)<<endl;
  cout<<linmaps1(A)<<endl;
  cout<<linmaps2(A)<<endl;
  cout<<"-----"<<endl;

  #ifdef _WITH_CUDA
  Ptensors1 Ag(A,1);
  cout<<linmaps0(Ag)<<endl;
  cout<<linmaps1(Ag)<<endl;
  cout<<linmaps2(Ag)<<endl;
  #endif

}
