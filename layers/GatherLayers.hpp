#ifndef _ptens_GatherLayers
#define _ptens_GatherLayers

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Hgraph.hpp"


namespace ptens{

  void add_gather(Ptensors0& r, const Ptensors0& x, const Hgraph& G){
    PTENS_ASSRT(G.n==r.size());
    PTENS_ASSRT(G.m==x.size());
    G.forall_edges([&](const int i, const int j, const float v){
	r.view_of_tensor(i).add(x.view_of_tensor(j),v);
      });
  }

  Ptensors0 gather(const Ptensors0& x, const Hgraph& G){
    Ptensors0 R=Ptensors0::zero(G.n,x.get_nc());
    add_gather(R,x,G);
    return R;
  }

}

#endif 
