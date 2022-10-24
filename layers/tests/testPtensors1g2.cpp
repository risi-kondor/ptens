#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"
#include "GatherLayers.hpp"


using namespace ptens;
using namespace cnine;


template<typename TYPE>
Ptensors1 backward_linmap(const Ptensors1& x, const TYPE& g){
  Ptensors1 r=Ptensors1::zeros_like(x);
  add_linmaps_back(r,g);
  return r;
}

template<typename TYPE>
Ptensors1 backward_unite(const Ptensors1& x, const TYPE& g, const Hgraph& G){
  Ptensors1 r=Ptensors1::zeros_like(x);
  add_msg_back(r,g,G);
  return r;
}


int main(int argc, char** argv){

  cnine_session session;
  #ifdef _WITH_CUDA

  Ptensors1 A=Ptensors1::randn({{1,2,3},{3,5},{2}},2);
  Ptensors1 Ag(A,1);
  cout<<A<<endl;

  Hgraph G=Hgraph::randomd(3,0.3);
  cout<<G<<endl;

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
    Ptensors1 G=Ptensors1::randn_like(B);
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


  {
    auto B=unite1(A,G);
    cout<<"unite1:"<<B.diff2(unite1(Ag,G))<<endl;
    Ptensors1 g=Ptensors1::gaussian_like(B);
    auto Aback=backward_linmap(A,g,G);
    auto Abackg=backward_linmap(Ag,g.to_device(1),G);
    cout<<Aback.diff2(Abackg)<<endl;
  }



  #endif

}
