#ifndef _ptens_GatherLayers
#define _ptens_GatherLayers

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Hgraph.hpp"


namespace ptens{

  #ifdef _WITH_CUDA
  extern void Ptensors0_gather_cu(cnine::RtensorPack& R,const cnine::RtensorPack& x, const cnine::CSRmatrix<float>& gmap, const cudaStream_t& stream);
  #endif 


  void add_gather(Ptensors0& r, const Ptensors0& x, const Hgraph& G){
    PTENS_ASSRT(G.n==r.size());
    PTENS_ASSRT(G.m==x.size());
    if(r.dev==0){
      G.forall_edges([&](const int i, const int j, const float v){
	  r.view_of_tensor(i).add(x.view_of_tensor(j),v);
	});
    }
    if(r.dev==1) CUDA_STREAM(Ptensors0_gather_cu(r,x,G.get_gmap(),stream));
  }

  Ptensors0 gather(const Ptensors0& x, const Hgraph& G){
    Ptensors0 R=Ptensors0::zero(G.n,x.get_nc(),x.dev);
    add_gather(R,x,G);
    return R;
  }

}

#endif 
