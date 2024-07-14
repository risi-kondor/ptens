/*
This file is part of ptens, a C++/CUDA library for permutation 
equivariant message passing. 
 
Copyright (c) 2023, Imre Risi Kondor

This source code file is subject to the terms of the noncommercial 
license distributed with cnine in the file LICENSE.TXT. Commercial 
use is prohibited. All redistributed versions of this file (in 
original or modified form) must retain this copyright notice and 
must be accompanied by a verbatim copy of the license. 
*/

#ifndef _Ptensors0_cu
#define _Ptensors0_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "Ptensors0.hpp"
#include "AindexPack.hpp"


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  rarr[rdir[2*i]+c]+=xarr[xdir[2*i]+c];
}


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int src=xiarr[xidir[2*b]];
  rarr[rdir[2*b]+c]+=xarr[xdir[2*src]+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  xarr[xdir[2*i]+c]+=rarr[rdir[2*i]+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir, const int* bmap){
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];

  float t=0;
  for(int j=0; j<N; j++){
    const int src=bmap[boffs+2*j];
    const float w=*reinterpret_cast<const float*>(bmap+boffs+2*j+1);
    t+=w*rarr[rdir[2*src]+c];
  }
  xarr[xdir[2*target]+c]+=t;
}


__global__ void Ptensors0_gather_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* marr, const int* mdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;

  const int moffs=mdir[2*i];
  const int N=mdir[2*i+1]/2;
  float t=0;
  for(int j=0; j<N; j++){
    const int jix=*reinterpret_cast<const int*>(marr+moffs+2*j);
    const int jweight=marr[moffs+2*j+1];
    t+=jweight*xarr[xdir[2*jix]+c];
  }
  rarr[rdir[2*i]+c]+=t;
}




// -----------------------------------------------------------------------------------------------------------


namespace ptens{

  void Ptensors0_reduce0_cu(Ptensors0& R, const Ptensors0& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    if(R.size()==0) return;
    Ptensors0_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors0_reduce0_cu(Ptensors0& R, const Ptensors0& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    const_cast<AindexPack&>(list).to_device(1);
    if(R.size()==0) return;
    Ptensors0_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }

  void Ptensors0_broadcast0_cu(Ptensors0& x, const Ptensors0& R, const int offs, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    if(R.size()==0) return;
    Ptensors0_broadcast0_kernel<<<R.size(),x.nc,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors0_broadcast0_cu(Ptensors0& x, const Ptensors0& R, const AindexPack& list, const int offs, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    if(list.get_bmap().n==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    Ptensors0_broadcast0_kernel<<<list.get_bmap().n,R.nc,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }



}

#endif 
