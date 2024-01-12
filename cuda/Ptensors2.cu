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

#ifndef _Ptensors2_cu
#define _Ptensors2_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPackB.hpp"
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


// ---- Reduce -----------------------------------------------------------------------------------------------


__global__ void Ptensors2_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]; // changed 

  const float* x=xarr+xdir[4*q]+c;
  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(i*k+j)*nc];
  rarr[rdir[2*q]+c]+=t;
  t=0;
  for(int i=0; i<k; i++)
    t+=x[(i*(k+1))*nc];
  rarr[rdir[2*q]+c+nc]+=t;
}

__global__ void Ptensors2_reduce0n_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]; // changed 

  const float* x=xarr+xdir[4*q]+c;
  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(i*k+j)*nc];
  rarr[rdir[2*q]+c]+=t/(k*k);
  t=0;
  for(int i=0; i<k; i++)
    t+=x[(i*(k+1))*nc];
  rarr[rdir[2*q]+c+nc]+=t/k;
}


// contracting version
__global__ void Ptensors2_reduce0B_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]; // changed 
  const int rnc=rdir[2*q+1];

  const float* x=xarr+xdir[4*q]+c;
  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(i*k+j)*nc];
  rarr[rdir[2*q]+c]+=t;
  t=0;
  for(int i=0; i<k; i++)
    t+=x[(i*(k+1))*nc+rnc];
  rarr[rdir[2*q]+c]+=t;
}


__global__ void Ptensors2_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,b);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
  rarr[rdir[2*b]+c]+=t;
  t=0;
  for(int i=0; i<k; i++)
    t+=x[(ix[i+1]*(_k+1))*nc];
  rarr[rdir[2*b]+c+nc]+=t;
}


__global__ void Ptensors2_reduce0n_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,b);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
  rarr[rdir[2*b]+c]+=t/(k*k);
  t=0;
  for(int i=0; i<k; i++)
    t+=x[(ix[i+1]*(_k+1))*nc];
  rarr[rdir[2*b]+c+nc]+=t/k;
}


// contracting version
__global__ void Ptensors2_reduce0B_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,b);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float t=0;
  for(int i=0; i<k; i++){
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
    t+=x[(ix[i+1]*(_k+1))*nc+n];
  }  
  rarr[rdir[2*b]+c]+=t;
}




__global__ void Ptensors2_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]; // changed
  const int rnc=rdir[3*q+2]; // changed

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(j*k+i)*nc];
    r[i*rnc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(i*k+j)*nc];
    r[i*rnc+nc]+=t;
  }
  for(int i=0; i<k; i++)
    r[i*rnc+2*nc]+=x[i*(k+1)*nc];
}


__global__ void Ptensors2_reduce1n_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]; // changed
  const int rnc=rdir[3*q+2]; // changed

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(j*k+i)*nc];
    r[i*rnc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(i*k+j)*nc];
    r[i*rnc+nc]+=t/k;
  }
  for(int i=0; i<k; i++)
    r[i*rnc+2*nc]+=x[i*(k+1)*nc];
}


// contracting version
__global__ void Ptensors2_reduce1B_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]; // changed
  const int rnc=rdir[3*q+2]; // changed

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(j*k+i)*nc];
    r[i*rnc]+=t;
  }
  x+=rnc;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(i*k+j)*nc];
    r[i*rnc]+=t;
  }
  x+=rnc;
  for(int i=0; i<k; i++)
    r[i*rnc]+=x[i*(k+1)*nc];
}


__global__ void Ptensors2_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,b);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  const int rnc=rdir[2];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[3*b]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[j+1]*_k+ix[i+1])*nc];
    r[i*rnc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
    r[i*rnc+nc]+=t;
  }
  for(int i=0; i<k; i++)
    r[i*rnc+2*nc]+=x[ix[i+1]*(_k+1)*nc];
}


__global__ void Ptensors2_reduce1n_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,b);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  const int rnc=rdir[2];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[3*b]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[j+1]*_k+ix[i+1])*nc];
    r[i*rnc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
    r[i*rnc+nc]+=t/k;
  }
  for(int i=0; i<k; i++)
    r[i*rnc+2*nc]+=x[ix[i+1]*(_k+1)*nc];
}


// contracting version
__global__ void Ptensors2_reduce1B_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,b);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  const int rnc=rdir[2];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[3*b]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[j+1]*_k+ix[i+1])*nc];
    r[i*rnc]+=t;
  }
  x+=n;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
    r[i*rnc]+=t;
  }
  x+=n;
  for(int i=0; i<k; i++)
    r[i*rnc]+=x[ix[i+1]*(_k+1)*nc];
}




__global__ void Ptensors2_reduce2_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[4*q+3];

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*rnc]+=x[(i*k+j)*nc];
}


// contracting version and flipping
__global__ void Ptensors2_reduce2B_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[4*q+3];

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*rnc]+=x[(i*k+j)*nc]+x[(j*k+i)*nc+rnc];
}


__global__ void Ptensors2_reduce2_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  const int rnc=rdir[3];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*rnc]+=x[(ix[i+1]*_k+ix[j+1])*nc];
}


// contracting version and flipping
__global__ void Ptensors2_reduce2B_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  __syncthreads();
  const int _k=xdir[4*ix[0]+1];
  const int nc=xdir[3];
  const int rnc=rdir[3];
  if(c>=n) return;

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*rnc]+=x[(ix[i+1]*_k+ix[j+1])*nc]+x[(ix[j+1]*_k+ix[i+1])*nc+n];
}



// ---- Broadcast --------------------------------------------------------------------------------------------


__global__ void Ptensors2_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[2*q+1];
  //if(c>=rnc) return; // change elsewhere too!

  float* x=xarr+xdir[4*q]+c;
  const float t=rarr[rdir[2*q]+c];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=t;
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc+rnc]+=t;
}


// contracting version
__global__ void Ptensors2_broadcast0B_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  //const int rnc=rdir[2*q+1];
  //if(c>=nc) return; // change elsewhere too!

  float* x=xarr+xdir[4*q]+c;
  float t=rarr[rdir[2*q]+c];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=t;
  t=rarr[rdir[2*q]+nc+c];
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc]+=t;
}


// contracting version
__global__ void Ptensors2_broadcast0Bn_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  //const int rnc=rdir[2*q+1];
  //if(c>=nc) return; // change elsewhere too!

  float* x=xarr+xdir[4*q]+c;
  float t=rarr[rdir[2*q]+c]/(k*k);
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=t;
  t=rarr[rdir[2*q]+nc+c]/k;
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc]+=t;
}


__global__ void Ptensors2_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  const int rnc=rdir[1];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=rnc) continue; 

    const float t=rarr[rdir[2*src]+c];
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=t;
    for(int i=0; i<k; i++)
      x[(ix[i+1]*(_k+1))*nc+rnc]+=t;
  }
}


// contracting version
__global__ void Ptensors2_broadcast0B_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  //const int rnc=rdir[1];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=nc) continue; 

    float t=rarr[rdir[2*src]+c];
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=t;
    t=rarr[rdir[2*src]+c+nc];
    for(int i=0; i<k; i++)
      x[(ix[i+1]*(_k+1))*nc]+=t;
  }
}


// contracting version
__global__ void Ptensors2_broadcast0Bn_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  //const int rnc=rdir[1];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=nc) continue; 

    float t=rarr[rdir[2*src]+c]/(k*k);
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=t;
    t=rarr[rdir[2*src]+c+nc]/k;
    for(int i=0; i<k; i++)
      x[(ix[i+1]*(_k+1))*nc]+=t;
  }
}




__global__ void Ptensors2_broadcast1_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[3*q+2];
  //if(c>=rnc) return; // redundant here

  float* x=xarr+xdir[4*q]+c;
  const float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=r[i*rnc];
    for(int j=0; j<k; j++)
      x[(j*k+i)*nc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=r[i*rnc];
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc+rnc]+=t;
  }
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc+2*rnc]+=r[i*rnc];
}


// contracting version
__global__ void Ptensors2_broadcast1B_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[3*q+2];
  //if(c>=nc) return; // redundant here

  float* x=xarr+xdir[4*q]+c;
  const float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=r[i*rnc];
    for(int j=0; j<k; j++)
      x[(j*k+i)*nc]+=t;
  }
  //r+=nc;
  for(int i=0; i<k; i++){
    float t=r[i*rnc+nc];
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=t;
  }
  //r+=nc;
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc]+=r[i*rnc+2*nc];
}

// contracting version
__global__ void Ptensors2_broadcast1Bn_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[3*q+2];
  //if(c>=nc) return; // redundant here

  float* x=xarr+xdir[4*q]+c;
  const float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=r[i*rnc]/k;
    for(int j=0; j<k; j++)
      x[(j*k+i)*nc]+=t;
  }
  //r+=nc;
  for(int i=0; i<k; i++){
    float t=r[i*rnc+nc]/k;
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=t;
  }
  //r+=nc;
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc]+=r[i*rnc+2*nc];
}


__global__ void Ptensors2_broadcast1_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  const int rnc=rdir[2];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=rnc) return;

    const float* r=rarr+rdir[3*src]+c;
    for(int i=0; i<k; i++){
      float t=r[i*rnc];
      for(int j=0; j<k; j++)
	x[(ix[j+1]*_k+ix[i+1])*nc]+=t;
    }
    for(int i=0; i<k; i++){
      float t=r[i*rnc];
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc+rnc]+=t;
    }
    for(int i=0; i<k; i++)
      x[ix[i+1]*(_k+1)*nc+2*rnc]+=r[i*rnc];
  }
}


// contracting version
__global__ void Ptensors2_broadcast1B_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  const int rnc=rdir[2];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=nc) return;

    const float* r=rarr+rdir[3*src]+c;
    for(int i=0; i<k; i++){
      float t=r[i*rnc];
      for(int j=0; j<k; j++)
	x[(ix[j+1]*_k+ix[i+1])*nc]+=t;
    }
    for(int i=0; i<k; i++){
      float t=r[i*rnc+nc];
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=t;
    }
    for(int i=0; i<k; i++)
      x[ix[i+1]*(_k+1)*nc]+=r[i*rnc+2*nc];
  }
}


// contracting version
__global__ void Ptensors2_broadcast1Bn_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  const int rnc=rdir[2];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=nc) return;

    const float* r=rarr+rdir[3*src]+c;
    for(int i=0; i<k; i++){
      float t=r[i*rnc]/k;
      for(int j=0; j<k; j++)
	x[(ix[j+1]*_k+ix[i+1])*nc]+=t;
    }
    for(int i=0; i<k; i++){
      float t=r[i*rnc+nc]/k;
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=t;
    }
    for(int i=0; i<k; i++)
      x[ix[i+1]*(_k+1)*nc]+=r[i*rnc+2*nc];
  }
}




__global__ void Ptensors2_broadcast2_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[4*q+3];
  //if(c>=rnc) return; // change elsewhere too!

  float* x=xarr+xdir[4*q]+c;
  const float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=r[(i*k+j)*rnc];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(j*k+i)*nc+rnc]+=r[(i*k+j)*rnc];
}


// contracting version and without flipping
__global__ void Ptensors2_broadcast2B_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[4*q+3];
  //if(c>=rnc) return; // change elsewhere too!

  float* x=xarr+xdir[4*q]+c;
  const float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=r[(i*k+j)*rnc];
}


__global__ void Ptensors2_broadcast2_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  const int rnc=rdir[3];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=rnc) return;

    const float* r=rarr+rdir[4*src]+c;
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=r[(i*k+j)*rnc];
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc+rnc]+=r[(j*k+i)*rnc];
  }
}


// contracting version and without flipping
__global__ void Ptensors2_broadcast2B_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, 
  const float* rarr, const int* rdir, const int* bmap){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];
  const int nc=xdir[3];
  const int rnc=rdir[3];

  float* x=xarr+xdir[4*target]+c;
  for(int s=0; s<N; s++){
    const int src=bmap[boffs+2*s];
    const int k=load_indices(ix,xiarr,xidir,src);
    const int _k=xdir[4*target+1];
    __syncthreads();
    if(c>=nc) return;

    const float* r=rarr+rdir[4*src]+c;
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	x[(ix[i+1]*_k+ix[j+1])*nc]+=r[(i*k+j)*rnc];
  }
}


// ---- Outer ------------------------------------------------------------------------------------------------

/*
__global__ void Ptensors2_outer20_kernel(float* rarr, const int* rdir, const float* xarr, const float* xdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];


  float* r=rarr+rdir[4*q]+c;
  const float* x=xarr+xdir[4*q]+c;
  const float* y=yarr+ydir[2*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*nc]+=x[(i*k+j)*nc];
}
*/


// ---- Outer -----------------------------------------------------------------------------------------------


__global__ void Ptensors2_add_outer20_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int yc=threadIdx.y;
  const int rc=xc*ydir[2*q+1]+yc;
  const int k=xdir[4*q+1];
  const int nxc=xdir[4*q+3];
  const int nrc=rdir[4*q+3];

  float* r=rarr+rdir[4*q]+rc;
  const float* x=xarr+xdir[4*q]+xc;
  const float t=yarr[ydir[2*q]+yc];
  for(int i=0; i<k*k; i++)
    r[i*nrc]+=t*x[i*nxc];
}


__global__ void Ptensors2_add_outer20_back0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int rc=xc*ydir[2*q+1];
  const int k=xdir[4*q+1];
  const int nxc=xdir[4*q+3];
  const int nyc=ydir[2*q+1];
  const int nrc=rdir[4*q+3];

  float* x=xarr+xdir[4*q]+xc;
  const float* r=rarr+rdir[4*q]+rc;
  const float* y=yarr+ydir[2*q];

  for(int i=0; i<k*k; i++){
    float t=0;
    for(int yc=0; yc<nyc; yc++)
      t+=r[i*nrc+yc]*y[yc];
    x[i*nxc]+=t;
  }
}


__global__ void Ptensors2_add_outer20_back1_kernel(float* yarr, const int* ydir, const float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int yc=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nxc=xdir[4*q+3];
  const int nyc=ydir[2*q+1];
  const int nrc=rdir[4*q+3];

  float t=0;
  for(int i=0; i<k*k; i++){
    const float* x=xarr+xdir[4*q]+i*nxc;
    const float* r=rarr+rdir[4*q]+i*nrc+yc;
    for(int xc=0; xc<nxc; xc++)
      t+=r[nyc*xc]*x[xc];
  }
  yarr[ydir[2*q]+yc]+=t;
}


__global__ void Ptensors2_add_outer02_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int yc=threadIdx.y;
  const int rc=xc*ydir[4*q+3]+yc;
  const int k=ydir[4*q+1];
  //const int nxc=xdir[2*q+1];
  const int nyc=ydir[4*q+3];
  const int nrc=rdir[4*q+3];

  float* r=rarr+rdir[4*q]+rc;
  const float* y=yarr+ydir[4*q]+yc;
  const float t=xarr[xdir[2*q]+xc];
  for(int i=0; i<k*k; i++)
    r[i*nrc]+=t*y[i*nyc];
}


__global__ void Ptensors2_add_outer02_back0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int rc=xc*ydir[4*q+3];
  const int k=ydir[4*q+1];
  //const int nxc=xdir[2*q+1];
  const int nyc=ydir[4*q+3];
  const int nrc=rdir[4*q+3];

 float t=0;
 for(int i=0; i<k*k; i++){
   const float* y=yarr+ydir[4*q]+i*nyc;
   const float* r=rarr+rdir[4*q]+i*nrc+rc;
   for(int yc=0; yc<nyc; yc++)
     t+=r[yc]*y[yc];
 }
 xarr[xdir[2*q]+xc]+=t;
}


__global__ void Ptensors2_add_outer02_back1_kernel(float* yarr, const int* ydir, const float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int yc=threadIdx.x;
  const int k=ydir[4*q+1];
  const int nxc=xdir[2*q+1];
  const int nyc=ydir[4*q+3];
  const int nrc=rdir[4*q+3];

  float* y=yarr+ydir[4*q]+yc;
  const float* r=rarr+rdir[4*q]+yc;
  const float* x=xarr+xdir[2*q];

  for(int i=0; i<k*k; i++){
    float t=0;
    for(int xc=0; xc<nxc; xc++)
      t+=r[i*nrc+xc*nyc]*x[xc];
    y[i*nyc]+=t;
  }
   
}


__global__ void Ptensors2_add_outer11_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int yc=threadIdx.y;
  const int rc=xc*ydir[3*q+2]+yc;
  const int k=ydir[3*q+1];
  const int nxc=xdir[3*q+2];
  const int nyc=ydir[3*q+2];
  const int nrc=rdir[4*q+3];

  float* r=rarr+rdir[4*q]+rc;
  const float* x=xarr+xdir[3*q]+xc;
  const float* y=yarr+ydir[3*q]+yc;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*nrc]+=x[i*nxc]*y[j*nyc];
}


__global__ void Ptensors2_add_outer11_back0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir, const float* yarr, const int* ydir){
  const int q=blockIdx.x;
  const int xc=threadIdx.x;
  const int rc=xc*ydir[3*q+2];
  const int k=ydir[3*q+1];
  const int nxc=xdir[2*q+2];
  const int nyc=ydir[3*q+2];
  const int nrc=rdir[4*q+3];

 for(int i=0; i<k; i++){
   float t=0;
   for(int j=0; j<k; j++){
     const float* y=yarr+ydir[3*q]+j*nyc;
     const float* r=rarr+rdir[4*q]+(i*k+j)*nrc+rc;
     for(int yc=0; yc<nyc; yc++)
       t+=r[yc]*y[yc];
   }
   xarr[xdir[3*q]+i*nxc+xc]+=t;
 }
}


__global__ void Ptensors2_add_outer11_back1_kernel(float* yarr, const int* ydir, const float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int yc=threadIdx.x;
  const int k=ydir[3*q+1];
  const int nxc=xdir[3*q+2];
  const int nyc=ydir[3*q+2];
  const int nrc=rdir[4*q+3];

  for(int j=0; j<k; j++){
    float t=0;
    for(int i=0; i<k; i++){
      const float* x=xarr+xdir[3*q]+i*nxc;
      const float* r=rarr+rdir[4*q]+(i*k+j)*nrc+yc;
      for(int xc=0; xc<nxc; xc++)
	t+=r[xc*nyc]*x[xc];
    }
    yarr[ydir[3*q]+j*nyc+yc]+=t;
  }
   
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{


  void Ptensors2_reduce0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce0n_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce0B_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce0B_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce0_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }

  void Ptensors2_reduce0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce0n_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }

  void Ptensors2_reduce0B_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce0B_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }



  void Ptensors2_reduce1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce1_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce1n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce1n_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce1B_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce1B_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce1_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }

  void Ptensors2_reduce1n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce1n_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }

  void Ptensors2_reduce1B_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce1B_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }



  void Ptensors2_reduce2_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce2_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce2B_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce2B_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce2_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce2_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }

  void Ptensors2_reduce2B_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce2B_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),n);
  }



  void Ptensors2_broadcast0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.nc;
    Ptensors2_broadcast0_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast0B_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=x.nc;
    Ptensors2_broadcast0B_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast0Bn_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=x.nc;
    Ptensors2_broadcast0Bn_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32);
    Ptensors2_broadcast0_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast0B_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32); // should be x.nc??
    Ptensors2_broadcast0B_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast0Bn_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32); // should be x.nc??
    Ptensors2_broadcast0Bn_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }



  void Ptensors2_broadcast1_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.nc;
    Ptensors2_broadcast1_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast1B_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.nc; // is this correct?
    Ptensors2_broadcast1B_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast1Bn_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.nc;
    Ptensors2_broadcast1Bn_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast1_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32); // change dim_of?
    Ptensors2_broadcast1_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast1B_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32);
    Ptensors2_broadcast1B_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast1Bn_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32);
    Ptensors2_broadcast1Bn_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }


  void Ptensors2_broadcast2_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.nc;
    Ptensors2_broadcast2_kernel<<<R.size(),n,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast2B_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=x.nc;
    Ptensors2_broadcast2B_kernel<<<R.size(),n,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast2_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32);
    Ptensors2_broadcast2_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast2B_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.nc,list.max_nix()+1),32);
    Ptensors2_broadcast2B_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.get_arrg(),list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }


  void Ptensors2_add_outer20_cu(cnine::RtensorPackB& r, const cnine::RtensorPackB& x, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    dim3 threads(x.nc,y.nc);
    Ptensors2_add_outer20_kernel<<<r.size(),threads,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors2_add_outer20_back0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& r, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    Ptensors2_add_outer20_back0_kernel<<<r.size(),x.nc,0,stream>>>
      (x.arrg,x.dir.garr(dev),r.arrg,r.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors2_add_outer20_back1_cu(cnine::RtensorPackB& y, const cnine::RtensorPackB& r, const cnine::RtensorPackB& x, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    Ptensors2_add_outer20_back1_kernel<<<r.size(),y.nc,0,stream>>>
      (y.arrg,y.dir.garr(dev),r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev));
  }



  void Ptensors2_add_outer02_cu(cnine::RtensorPackB& r, const cnine::RtensorPackB& x, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    dim3 threads(x.nc,y.nc);
    Ptensors2_add_outer02_kernel<<<r.size(),threads,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors2_add_outer02_back0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& r, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    Ptensors2_add_outer02_back0_kernel<<<r.size(),x.nc,0,stream>>>
      (x.arrg,x.dir.garr(dev),r.arrg,r.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors2_add_outer02_back1_cu(cnine::RtensorPackB& y, const cnine::RtensorPackB& r, const cnine::RtensorPackB& x, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    Ptensors2_add_outer02_back1_kernel<<<r.size(),y.nc,0,stream>>>
      (y.arrg,y.dir.garr(dev),r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev));
  }


  void Ptensors2_add_outer11_cu(cnine::RtensorPackB& r, const cnine::RtensorPackB& x, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    dim3 threads(x.nc,y.nc);
    Ptensors2_add_outer11_kernel<<<r.size(),threads,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors2_add_outer11_back0_cu(cnine::RtensorPackB& x, const cnine::RtensorPackB& r, const cnine::RtensorPackB& y, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    Ptensors2_add_outer11_back0_kernel<<<r.size(),x.nc,0,stream>>>
      (x.arrg,x.dir.garr(dev),r.arrg,r.dir.garr(dev),y.arrg,y.dir.garr(dev));
  }

  void Ptensors2_add_outer11_back1_cu(cnine::RtensorPackB& y, const cnine::RtensorPackB& r, const cnine::RtensorPackB& x, 
    const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(y.dev==1);
    Ptensors2_add_outer11_back1_kernel<<<r.size(),y.nc,0,stream>>>
      (y.arrg,y.dir.garr(dev),r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev));
  }


}


#endif 
