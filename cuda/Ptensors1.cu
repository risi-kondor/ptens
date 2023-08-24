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
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPack.hpp"
#include "AindexPack.hpp"


__forceinline__ __device__ int load_indices(int* ix, const int* xiarr, const int* xidir, const int q){
  int offs=xidir[2*q];
  int n=xidir[2*q+1];
  int t=threadIdx.x;
  if(t<n){
    ix[t]=xiarr[offs+t];
  }
  return n-1;
}


// ---- mprod ------------------------------------------------------------------------------------------------


__global__ void Ptensors1_add_mprod(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* yarr){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int ncx=xdir[3*q+2];
  const int ncr=rdir[3*q+2];

  const float* x=xarr+xdir[3*q];
  float* r=rarr+rdir[3*q];

  for(int i=0; i<k; i++){
    float t=0;
    const float* xrow=x+i*ncx;
    const float* ycol=yarr+c;
    for(int j=0; j<ncx; j++)
      t+=xrow[j]*ycol[j*ncr];
    r[i*ncr+c]+=t;
  }
}


/*
__global__ void Ptensors1_add_mprod_back0(float* rarr, const int* rdir, const float* rarr, const int* rdir, const float* yarr){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int ncx=xdir[3*q+2];
  const int ncr=rdir[3*q+2];

  const float* x=xarr+xdir[3*q];
  const float* r=rarr+rdir[3*q];

  for(int i=0; i<k; i++){
    float t=0;
    float* xrow=x+i*ncx;
    float* yrow=yarr+c*ncx;
    for(int j=0; j<ncx; j++)
      t+=xrow[j]*yrow[j];
    r[i*ncr+c]+=t;
  }
}


__global__ void Ptensors1_add_mprod_back1(float* rarr, const int* rdir, const float* rarr, const int* rdir, const float* yarr){
}
*/

// ---- Reduce -----------------------------------------------------------------------------------------------


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


__global__ void Ptensors1_reduce0n_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];
  //if(c>=nc) return;

  const float* x=xarr+xdir[3*q]+c;
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[i*nc];
  rarr[rdir[2*q]+c]+=t/k;
}


__global__ void Ptensors1_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  __syncthreads();
  const int nc=xdir[2];
  if(c>=n) return;

  const float* x=xarr+xdir[3*ix[0]]+c;
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[ix[i+1]*nc];
  rarr[rdir[2*q]+c]+=t;
}


__global__ void Ptensors1_reduce0n_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  __syncthreads();
  const int nc=xdir[2];
  if(c>=n) return;

  const float* x=xarr+xdir[3*ix[0]]+c;
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[ix[i+1]*nc];
  rarr[rdir[2*q]+c]+=t/k;
}


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


__global__ void Ptensors1_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  __syncthreads();
  const int nc=xdir[2];
  const int rnc=rdir[2];
  if(c>=n) return;

  const float* x=xarr+xdir[3*ix[0]]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++)
    r[i*rnc]+=x[ix[i+1]*nc];
}


// ---- Broadcast --------------------------------------------------------------------------------------------


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


__global__ void Ptensors1_broadcast0n_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];
  const int rnc=rdir[2*q+1];
  if(c>=rnc) return;

  float* x=xarr+xdir[3*q]+c;
  const float t=rarr[rdir[2*q]+c]/k;
  for(int i=0; i<k; i++)
    x[i*nc]+=t;
}


__global__ void Ptensors1_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  //if(c==0) printf("%d target=%d\n",b,target);

  const int nc=xdir[2]; //xdir[3*target+2];
  const int rnc=rdir[1]; //xdir[3*target+2];


  float* x=xarr+xdir[3*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    //if(c==0) printf("%d %d %d\n",b,s,src);
    const int k=load_indices(ix,xiarr,xidir,src);
    __syncthreads();
    if(c>=rnc) continue;
    float t=rarr[rdir[2*src]+c];
    //if(c==0) printf("%d %d %d %f\n",b,s,src,t);
    for(int i=0; i<k; i++){
      x[ix[i+1]*nc]+=t;
    }
  }

  return;
}


__global__ void Ptensors1_broadcast0n_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  //if(c==0) printf("%d target=%d\n",b,target);

  const int nc=xdir[2]; //xdir[3*target+2];
  const int rnc=rdir[1]; //xdir[3*target+2];


  float* x=xarr+xdir[3*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    //if(c==0) printf("%d %d %d\n",b,s,src);
    const int k=load_indices(ix,xiarr,xidir,src);
    __syncthreads();
    if(c>=rnc) continue;
    float t=rarr[rdir[2*src]+c]/k;
    //if(c==0) printf("%d %d %d %f\n",b,s,src,t);
    for(int i=0; i<k; i++){
      x[ix[i+1]*nc]+=t;
    }
  }

  return;
}


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


__global__ void Ptensors1_broadcast1_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  //if(c==0) printf("target=%d\n",target);

  const int nc=xdir[2]; //xdir[3*target+2];
  const int rnc=rdir[2]; //xdir[3*target+2];

  float* x=xarr+xdir[3*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    __syncthreads();
    //if(c>=rnc) return;
    if(c>=rnc) continue; // changed 

    const float* r=rarr+rdir[3*src]+c;
    for(int i=0; i<k; i++){
      //if(c==0) printf("%d %d %d %d %d %d %d\n",src,target,i,ix[i+1],xdir[3*target],xdir[3*target+1],nc);
      x[ix[i+1]*nc]+=r[i*rnc];
    }
  }
}


// ---- Outer -----------------------------------------------------------------------------------------------


__global__ void Ptensors1_add_outer10_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int yc=threadIdx.y;
  const int rc=xc*ydir[2*q+1]+yc;
  const int k=xdir[3*q+1];
  const int nxc=xdir[3*q+2];
  const int nrc=rdir[3*q+2];

  float* r=rarr+rdir[3*q]+rc;
  const float* x=xarr+xdir[3*q]+xc;
  const float t=yarr[ydir[2*q]+yc];
  for(int i=0; i<k; i++)
    r[i*nrc]+=t*x[i*nxc];
}


__global__ void Ptensors1_add_outer10_back0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int rc=xc*ydir[2*q+1];
  const int k=xdir[3*q+1];
  const int nxc=xdir[3*q+2];
  const int nyc=ydir[2*q+1];
  const int nrc=rdir[3*q+2];

  float* x=xarr+xdir[3*q]+xc;
  const float* r=rarr+rdir[3*q]+rc;
  const float* y=yarr+ydir[2*q];

  for(int i=0; i<k; i++){
    float t=0;
    for(int yc=0; yc<nyc; yc++)
      t+=r[i*nrc+yc]*y[yc];
    x[i*nxc]+=t;
  }
}


__global__ void Ptensors1_add_outer10_back1_kernel(float* yarr, const int* ydir, const float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int yc=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nxc=xdir[3*q+2];
  const int nyc=ydir[2*q+1];
  const int nrc=rdir[3*q+2];

  float t=0;
  for(int i=0; i<k; i++){
    const float* x=xarr+xdir[3*q]+i*nxc;
    const float* r=rarr+rdir[3*q]+i*nrc+yc;
    for(int xc=0; xc<nxc; xc++)
      t+=r[nyc*xc]*x[xc];
  }
  yarr[ydir[2*q]+yc]+=t;
  
}

__global__ void Ptensors1_add_outer01_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int yc=threadIdx.y;
  const int rc=xc*ydir[3*q+2]+yc;
  const int k=ydir[3*q+1];
  //const int nxc=xdir[2*q+1];
  const int nyc=ydir[3*q+2];
  const int nrc=rdir[3*q+2];

  float* r=rarr+rdir[3*q]+rc;
  const float* y=yarr+ydir[3*q]+yc;
  const float t=xarr[xdir[2*q]+xc];
  for(int i=0; i<k; i++)
    r[i*nrc]+=t*y[i*nyc];
}


__global__ void Ptensors1_add_outer01_back0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int rc=xc*ydir[3*q+2];
  const int k=ydir[3*q+1];
  //const int nxc=xdir[2*q+1];
  const int nyc=ydir[3*q+2];
  const int nrc=rdir[3*q+2];

 float t=0;
  for(int i=0; i<k; i++){
    const float* y=yarr+ydir[3*q]+i*nyc;
    const float* r=rarr+rdir[3*q]+i*nrc+rc;
    for(int yc=0; yc<nyc; yc++)
      t+=r[yc]*y[yc];
  }
  xarr[xdir[2*q]+xc]+=t;

}


__global__ void Ptensors1_add_outer01_back1_kernel(float* yarr, const int* ydir, const float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int yc=threadIdx.x;
  const int k=ydir[3*q+1];
  const int nxc=xdir[2*q+1];
  const int nyc=ydir[3*q+2];
  const int nrc=rdir[3*q+2];

  float* y=yarr+ydir[3*q]+yc;
  const float* r=rarr+rdir[3*q]+yc;
  const float* x=xarr+xdir[2*q];

  for(int i=0; i<k; i++){
    float t=0;
    for(int xc=0; xc<nxc; xc++)
      t+=r[i*nrc+xc*nyc]*x[xc];
    y[i*nyc]+=t;
  }
   
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{


  void Ptensors1_reduce0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    Ptensors1_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors1_reduce0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    Ptensors1_reduce0n_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors1_reduce0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(list.size()==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    PTENS_ASSRT(list.dev==1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors1_reduce0_kernel<<<list.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }

  void Ptensors1_reduce0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(list.size()==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    PTENS_ASSRT(list.dev==1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors1_reduce0n_kernel<<<list.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }



  void Ptensors1_reduce1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    Ptensors1_reduce1_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors1_reduce1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(list.size()==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    PTENS_ASSRT(list.dev==1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors1_reduce1_kernel<<<list.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }



  void Ptensors1_broadcast0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    int n=R.nc; //dim_of(0,0);
    Ptensors1_broadcast0_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors1_broadcast0n_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    int n=R.nc; //dim_of(0,0);
    Ptensors1_broadcast0n_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors1_broadcast0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(list.get_bmap().n==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    PTENS_ASSRT(list.dev==1);
    int n=cnine::roundup(std::max(R.nc/*dim_of(0,0)*/,list.max_nix()+1),32);
    Ptensors1_broadcast0_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>> 
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1)); // 32 or 128
  }

  void Ptensors1_broadcast0n_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(list.get_bmap().n==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    PTENS_ASSRT(list.dev==1);
    int n=cnine::roundup(std::max(R.nc/*dim_of(0,0)*/,list.max_nix()+1),32);
    Ptensors1_broadcast0n_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>> 
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1)); // 32 or 128
  }



  void Ptensors1_broadcast1_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(R.size()==0) return;
    int n=R.nc; //dim_of(0,1);
    Ptensors1_broadcast1_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors1_broadcast1_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    if(list.get_bmap().n==0) return;
    const_cast<AindexPack&>(list).to_device(1);
    PTENS_ASSRT(list.dev==1);
    int n=cnine::roundup(std::max(R.nc/*dim_of(0,1)*/,list.max_nix()+1),32); // here??
    Ptensors1_broadcast1_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }



  void Ptensors1_add_outer10_cu(cnine::RtensorPackB& r, const cnine::RtensorPackB& x, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    if(r.size()==0) return;
    dim3 threads(x.dim_of(0,1),y.dim_of(0,0));
    Ptensors1_add_outer10_kernel<<<r.size(),threads,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors1_add_outer10_back0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& r, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    if(r.size()==0) return;
    Ptensors1_add_outer10_back0_kernel<<<r.size(),x.dim_of(0,1),0,stream>>>
      (x.arrg,x.dir.garr(dev),r.arrg,r.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors1_add_outer10_back1_cu(cnine::RtensorPackB& y, const cnine::RtensorPackB& r, const cnine::RtensorPackB& x, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    if(r.size()==0) return;
    Ptensors1_add_outer10_back1_kernel<<<r.size(),y.dim_of(0,0),0,stream>>>
      (y.arrg,y.dir.garr(dev),r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev));
  }



  void Ptensors1_add_outer01_cu(cnine::RtensorPackB& r, const cnine::RtensorPackB& x, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    if(r.size()==0) return;
    dim3 threads(x.dim_of(0,0),y.dim_of(0,1));
    Ptensors1_add_outer01_kernel<<<r.size(),threads,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors1_add_outer01_back0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& r, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    if(r.size()==0) return;
    Ptensors1_add_outer01_back0_kernel<<<r.size(),x.dim_of(0,0),0,stream>>>
      (x.arrg,x.dir.garr(dev),r.arrg,r.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors1_add_outer01_back1_cu(cnine::RtensorPackB& y, const cnine::RtensorPackB& r, const cnine::RtensorPackB& x, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    if(r.size()==0) return;
    Ptensors1_add_outer01_back1_kernel<<<r.size(),y.dim_of(0,1),0,stream>>>
      (y.arrg,y.dir.garr(dev),r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev));
  }


}


#endif 
