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
#include "Ltensor.hpp"
#include "AindexPackB.hpp"


typedef cnine::Ltensor<float> TENSOR;
typedef cnine::Ltensor<int> ITENSOR;


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int rs, const float* xarr, const int xs, 
  const int* map, const int maps, const int n){
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  rarr[map[b*maps]*rs+c]+=xarr[map[b*maps+2]*xs+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* rarr, const int rs, const float* xarr, const int xs, 
  const int* map, const int maps, const int* bmap){
  //extern __shared__ unsigned char _shared[]; 
  //int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[b+1];
  const int N=bmap[b+2]-bmap[b+1]-1;
  if(N==0) return;
  //const int target=bmap[3*b+2];

  float t=0;
  int target=0;
  for(int s=0; s<N; s++){
    const int row=bmap[boffs+s+1];
    //if(c<maps) ix[c]=map[row*maps+c];
    //__syncthreads();

    if(s==0) target=map[row*maps+2];
    t+=xarr[map[row*maps]*xs+c];
  }
  rarr[target*rs+c]+=t;
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{


  void Ptensors0_reduce0_cu(const TENSOR& R, const TENSOR& x, const AindexPackB& map, int offs, int n, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    if(map.dim(0)==0) return;
    Ptensors0_reduce0_kernel<<<map.dim(0),n,0,stream>>>(R.get_arr(),R.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors0_broadcast0_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    int n=x.dim(1);
    int nthrd=n; //cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors0_broadcast0_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr());
  }



}

#endif 
/*
__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  rarr[rdir[2*i]+c]+=xarr[xdir[2*i]+c];
}
*/
/*
__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  xarr[xdir[2*i]+c]+=rarr[rdir[2*i]+c];
}
*/
/*
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
*/
  /*
  void Ptensors0_reduce0_cu(Ptensors0& R, const Ptensors0& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    if(R.size()==0) return;
    Ptensors0_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }
  */
  /*
  void Ptensors0_broadcast0_cu(Ptensors0& x, const Ptensors0& R, const int offs, const cudaStream_t& stream){
    int dev=R.get_dev();
    PTENS_ASSRT(R.get_dev()==1);
    PTENS_ASSRT(x.get_dev()==1);
    if(R.size()==0) return;
    Ptensors0_broadcast0_kernel<<<R.size(),x.nc,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }
  */
