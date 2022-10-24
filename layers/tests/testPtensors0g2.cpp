#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"
#include "GatherLayers.hpp"


using namespace ptens;
using namespace cnine;

template<typename TYPE>
Ptensors0 backward_linmap(const Ptensors0& x, const TYPE& G){
  Ptensors0 R=Ptensors0::zeros_like(x);
  add_linmaps_back(R,G);
  return R;
}


int main(int argc, char** argv){

  cnine_session session;
  #ifdef _WITH_CUDA

  Ptensors0 A=Ptensors0::randn(2,3);
  Ptensors0 Ag(A,1);
  //cout<<A<<endl;

  {
    auto B=linmaps0(A);
    cout<<"linmaps0:"<<B.diff2(linmaps0(Ag))<<endl;
    Ptensors0 G=Ptensors0::gaussian_like(B);
    auto Aback=backward_linmap(A,G);
    auto Abackg=backward_linmap(Ag,G.to_device(1));
    cout<<Aback.diff2(Abackg)<<endl;
  }

 {
    auto B=linmaps1(A);
    cout<<"linmaps1:"<<B.diff2(linmaps1(Ag))<<endl;
    Ptensors1 G=Ptensors1::gaussian_like(B);
    auto Aback=backward_linmap(A,G);
    auto Abackg=backward_linmap(Ag,G.to_device(1));
    cout<<Aback.diff2(Abackg)<<endl;
  }

 {
    auto B=linmaps2(A);
    cout<<"linmaps2:"<<B.diff2(linmaps2(Ag))<<endl;
    Ptensors2 G=Ptensors2::randn_like(B);
    auto Aback=backward_linmap(A,G);
    auto Abackg=backward_linmap(Ag,G.to_device(1));
    cout<<Aback.diff2(Abackg)<<endl;
  }

  #endif


}
