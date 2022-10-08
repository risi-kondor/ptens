#ifndef _Ptensors2_cu
#define _Ptensors2_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPack.hpp"
#include "Ptensors2.hpp"
//#include "Rtensor2_view.hpp"
//#include "Rtensor3_view.hpp"
//#include "Itensor1_view.hpp"
//#include "Itensor2_view.hpp"
//#include "CUDAhelpers.hpp"


__device__ void load_indices(float* ix, const int* xiarr, const int* xidir, const int q){
  int offs=xidir[2*q];
  int n=xidir[2*q+1];
  int t=threadIdx.x;
  if(t<n){
    ix[t]=xiarr[offs+t]
  }
  return n-1
}


// ---- Reduce -----------------------------------------------------------------------------------------------


__global__ void Ptensors2_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]/2; 

  const float* x=xarr+xdir[4*q]+c
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


__global__ void Ptensors2_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  extern __shared__ unsigned char _shared[]; 
  float* ix=reinterpret_cast<float*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  const int _k=xdir[4*q+1];
  const int nc=xdir[4*q+3]/2;
  if(c>=nc) return;
  __syncthreads();

  const float* x=xarr+xdir[4*ix[0]]+c;
  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*_k+ix[j+1])*nc];
  rarr[rdir[2*q]+c]+=t;
  for(int i=0; i<k; i++)
    t+=x[(ix[i+1]*(_k+1))*nc];
  rarr[rdir[2*q]+c+nc]+=t;
}


__global__ void Ptensors2_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3]/3;

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++) // other way round??
      t+=x[(i*k+j)*nc];
    r[i*nc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(j*k+i)*nc];
    r[i*nc+nc]+=t;
  }
  for(int i=0; i<k; i++)
    r[i*nc+2*nc]+=x[i*(k+1)*nc];
}


__global__ void Ptensors2_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  extern __shared__ unsigned char _shared[]; 
  float* ix=reinterpret_cast<float*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  const int _k=xdir[4*q+1];
  const int nc=xdir[4*q+3]/3;
  if(c>=nc) return;
  __syncthreads();

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[i+1]*k+ix[j+1])*nc];
    r[i*nc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[j+1]*k+ix[i+1])*nc];
    r[i*nc+nc]+=t;
  }
  for(int i=0; i<k; i++)
    r[i*nc+2*nc]+=x[ix[i+1]*(k+1)*nc];
}


__global__ void Ptensors2_reduce2_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];

  const float* x=xarr+xdir[4*q]+c;
  float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*nc]=x[(i*k+j)*nc];
}


__global__ void Ptensors2_reduce2_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  extern __shared__ unsigned char _shared[]; 
  float* ix=reinterpret_cast<float*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  const int _k=xdir[4*q+1];
  const int nc=xdir[4*q+3]/3;
  if(c>=nc) return;
  __syncthreads();

  const float* x=xarr+xdir[4*ix[0]]+c;
  float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*nc]+=x[(ix[i+1]*_k+ix[j+1])*nc];
}


// ---- Broadcast --------------------------------------------------------------------------------------------


__global__ void Ptensors2_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[2*q+1];

  float* x=xarr+xdir[4*q]+c;
  const float t=rarr[rdir[2*q]+c];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(i*k+j)*nc]+=t;
  for(int i=0; i<k; i++)
    x[i*(k+1)*nc+rnc]+=t;
}


__global__ void Ptensors2_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  const int _k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[2*q+1];
  if(c>=nc) return;
  __syncthreads();

  float* x=xarr+xdir[4*ix[0]]+c;
  const float t=rarr[rdir[2*q]+c];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(ix[i+1]*_k+ix[j+1])*nc]+=t;
  for(int i=0; i<k; i++)
    x[(ix[i+1]*(_k+1)*nc+rnc]+=t;
}


__global__ void Ptensors2_broadcast1_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[3*q+2];

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


__global__ void Ptensors2_broadcast1_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  const int _k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[3*q+2];
  if(c>=nc) return;
  __syncthreads();

  float* x=xarr+xdir[4*ix[0]]+c;
  const float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++){
    float t=r[i*rnc];
    for(int j=0; j<k; j++)
      x[(ix[j+1]*_k+ix[i+1])*nc]+=t;
  }
  for(int i=0; i<k; i++){
    float t=r[i*rnc];
    for(int j=0; j<k; j++)
      x[(ix[i+1]*k+ix[j+1])*nc+rnc]+=t;
  }
  for(int i=0; i<k; i++)
    x[ix[i+1]*(k+1)*nc+2*rnc]+=r[i*rnc];
}


__global__ void Ptensors2_broadcast2_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[4*q+3];

  float* x=xarr+xdir[4*q]+c;
  const float* r=rarr+rdir[4*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(j*k+i)*nc]+=r[(i*k+j)*rnc];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(j*k+i)*nc+rnc]+=r[(j*k+i)*rnc];
}


__global__ void Ptensors2_broadcast2_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xiarr,xidir,q);
  const int _k=xdir[4*q+1];
  const int nc=xdir[4*q+3];
  const int rnc=rdir[4*q+3];
  if(c>=nc) return;
  __syncthreads();

  float* x=xarr+xdir[4*ix[0]]+c;
  const float* r=rarr+rdir[3*q]+c;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(ix[j+1]*_k+ix[i+1])*nc]+=r[(i*k+j)*rnc];
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      x[(ix[j+1]*_k+ix[i+1])*nc+rnc]+=r[(j*k+i)*rnc];
}


// -----------------------------------------------------------------------------------------------------------

namespace ptens{

  void Ptensors2_reduce0_cu(RtensorPack& R, const RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce0_kernel<R.size(),n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg);
  }

  void Ptensors2_reduce0_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors2_reduce0_kernel<R.size(),std::max(n,32),32,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg);
  }

  void Ptensors2_reduce1_cu(RtensorPack& R, const RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce1_kernel<R.size(),n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg);
  }

  void Ptensors2_reduce1_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors2_reduce1_kernel<R.size(),std::max(n,32),32,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg);
  }

  void Ptensors2_reduce2_cu(RtensorPack& R, const RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_reduce2_kernel<R.size(),n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg);
  }

  void Ptensors2_reduce2_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors2_reduce2_kernel<R.size(),std::max(n,32),32,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg);
  }


  void Ptensors2_broadcast0_cu(RtensorPack& x, const RtensorPack& R, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_broadcast0_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors2_broadcast0_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast0_kernel<R.size(),std::max(n,32),32,stream>(x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors2_broadcast1_cu(RtensorPack& x, const RtensorPack& R, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_broadcast1_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors2_broadcast1_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast1_kernel<R.size(),std::max(n,32),32,stream>(x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors2_broadcast2_cu(RtensorPack& x, const RtensorPack& R, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors2_broadcast1_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors2_broadcast2_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast1_kernel<R.size(),std::max(n,32),32,stream>(x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg,R.arrg,r.dir.arrg);
  }

}


#endif 
