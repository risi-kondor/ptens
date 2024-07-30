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

#ifndef _Ptensors1_cu
#define _Ptensors1_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "Ptens_base.hpp"
#include "Ltensor.hpp"
#include "AindexPackB.hpp"


typedef cnine::Ltensor<float> TENSOR;
typedef cnine::Ltensor<int> ITENSOR;



// ---- Reduce -----------------------------------------------------------------------------------------------


__global__ void Ptensors1_reduce0_kernel(float* rarr, int rs, const float* xarr, int xs, const int* maparr, int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  const int k=ix[1];
  __syncthreads();

  if(c>=n) return;
  const float* x=xarr+ix[2]*xs+c;
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[ix[i+4]*xs];
  rarr[ix[0]*rs+c]+=t;
}


__global__ void Ptensors1_reduce1_kernel(float* rarr, int rs, const float* xarr, const int xs, 
  const int* maparr, const int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  const int k=ix[1];
  __syncthreads();

  if(c>=n) return;
  const float* x=xarr+ix[2]*xs+c;
  float* r=rarr+ix[0]*rs+c;
  for(int i=0; i<k; i++)
    r[i*rs]+=x[ix[i+4]*xs];
}


// ---- Broadcast --------------------------------------------------------------------------------------------


__global__ void Ptensors1_broadcast0_kernel(float* rarr, const int rs, 
  const float* xarr, const int xs, const int* maparr, const int maps, const int* bmap, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int b=blockIdx.x;
  assert(b<bmap[0]);
  const int c=threadIdx.x;
  const int boffs=bmap[b+1];
  const int N=bmap[b+2]-bmap[b+1]-1;
  //const int target=bmap[boffs];
  //float* r=rarr+target*rs+c;

  for(int s=0; s<N; s++){
    const int row=bmap[boffs+s+1];
    if(c<maps) ix[c]=maparr[row*maps+c];
    const int k=ix[1];
    __syncthreads();

    if(c>=n) continue;
    //assert(ix[2]==target);
    float t=xarr[ix[0]*xs+c];
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++){
      r[ix[i+4]*rs]+=t;
    }
  }
}


__global__ void Ptensors1_broadcast1_kernel(float* rarr, const int rs, 
  const float* xarr, const int xs, const int* maparr, const int maps, const int* bmap, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int b=blockIdx.x;
  assert(b<bmap[0]);
  const int c=threadIdx.x;
  const int boffs=bmap[b+1];
  const int N=bmap[b+2]-bmap[b+1]-1;
  //const int target=bmap[boffs];
  //float* r=rarr+target*rs+c;

  for(int s=0; s<N; s++){
    const int row=bmap[boffs+s+1];
    if(c<maps) ix[c]=maparr[row*maps+c];
    const int k=ix[1];
    __syncthreads();

    if(c>=n) continue;
    //assert(ix[2]==target);
    const float* x=xarr+ix[0]*xs+c;
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++){
      r[ix[i+4]*rs]+=x[i*xs];
    }
  }
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{


  void Ptensors1_reduce0_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    if(map.dim(0)==0) return;

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    Ptensors1_reduce0_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors1_reduce1_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)+1),32);
    Ptensors1_reduce1_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

   void Ptensors1_broadcast0_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    //PTENS_ASSRT(map.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors1_broadcast0_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

  void Ptensors1_broadcast1_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    //PTENS_ASSRT(map.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors1_broadcast1_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

}


// ----- Linmaps --------------------------------------------------------------------------------------------

/*
__global__ void Ptensors1_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];
  //if(c>=nc) return;

  const float* x=xarr+xdir[3*q]+c;
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[i*nc];
  rarr[rdir[2*q]+c]+=t;
}
*/

/*
__global__ void Ptensors1_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];
  const int rnc=rdir[3*q+2];
  //if(c>=nc) return;

  const float* x=xarr+xdir[3*q]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++)
    r[i*rnc]+=x[i*nc];
}
*/


/*
__global__ void Ptensors1_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];
  const int rnc=rdir[2*q+1];
  if(c>=rnc) return;

  float* x=xarr+xdir[3*q]+c;
  const float t=rarr[rdir[2*q]+c];
  for(int i=0; i<k; i++)
    x[i*nc]+=t;
}
*/


/*
__global__ void Ptensors1_broadcast1_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];
  const int rnc=rdir[3*q+2];
  if(c>=rnc) return;

  float* x=xarr+xdir[3*q]+c;
  const float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++)
    x[i*nc]+=r[i*rnc];
}
*/

namespace ptens{

  /*
  void Ptensors1_reduce0_cu(TENSOR& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    Ptensors1_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }
  */

  /*
  void Ptensors1_reduce1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    Ptensors1_reduce1_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }
  */

 /*
  void Ptensors1_broadcast0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    int n=R.nc; 
    Ptensors1_broadcast0_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }
  */

  /*
  void Ptensors1_broadcast1_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    int n=R.nc;
    Ptensors1_broadcast1_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }
  */

}
#endif 
