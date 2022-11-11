#ifndef _Ptensors2_cu
#define _Ptensors2_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPack.hpp"
#include "AindexPack.hpp"
//#include "Ptensors2.hpp"
//#include "Rtensor2_view.hpp"
//#include "Rtensor3_view.hpp"
//#include "Itensor1_view.hpp"
//#include "Itensor2_view.hpp"
//#include "CUDAhelpers.hpp"


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


// -----------------------------------------------------------------------------------------------------------

namespace ptens{



  void Ptensors2_reduce0_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce0_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce0B_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce0B_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce0_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce0_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }

  void Ptensors2_reduce0B_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce0B_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }



  void Ptensors2_reduce1_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce1_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce1B_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce1B_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce1_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce1_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }

  void Ptensors2_reduce1B_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce1B_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }



  void Ptensors2_reduce2_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce2_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce2B_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce2B_kernel<<<R.size(),n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors2_reduce2_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce2_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }

  void Ptensors2_reduce2B_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, 
    int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    const int nthrd=cnine::roundup(std::max(n,list.max_nix()+1),32);
    Ptensors2_reduce2B_kernel<<<R.size(),nthrd,(list.max_nix()+1)*4,stream>>>
      (R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),n);
  }



  void Ptensors2_broadcast0_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.dim_of(0,0);
    Ptensors2_broadcast0_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast0B_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=x.dim_of(0,2);
    Ptensors2_broadcast0B_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast0_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.dim_of(0,0),list.max_nix()+1),32);
    Ptensors2_broadcast0_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast0B_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.dim_of(0,0),list.max_nix()+1),32);
    Ptensors2_broadcast0B_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }



  void Ptensors2_broadcast1_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.dim_of(0,1);
    Ptensors2_broadcast1_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast1B_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.dim_of(0,1);
    Ptensors2_broadcast1B_kernel<<<R.size(),n,0,stream>>>(x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast1_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.dim_of(0,0),list.max_nix()+1),32);
    Ptensors2_broadcast1_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast1B_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.dim_of(0,0),list.max_nix()+1),32);
    Ptensors2_broadcast1B_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }


  void Ptensors2_broadcast2_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=R.dim_of(0,2);
    Ptensors2_broadcast2_kernel<<<R.size(),n,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast2B_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int n=x.dim_of(0,2);
    Ptensors2_broadcast2B_kernel<<<R.size(),n,0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors2_broadcast2_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.dim_of(0,0),list.max_nix()+1),32);
    Ptensors2_broadcast2_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors2_broadcast2B_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, 
    const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<AindexPack&>(list).to_device(1);
    int n=cnine::roundup(std::max(R.dim_of(0,0),list.max_nix()+1),32);
    Ptensors2_broadcast2B_kernel<<<list.get_bmap().n,n,(list.max_nix()+1)*4,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

}


#endif 
