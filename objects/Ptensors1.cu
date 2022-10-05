#ifndef _Ptensors1_cu
#define _Ptensors1_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPack.hpp"
#include "Ptensors1.hpp"
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


__global__ void Ptensors1_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];

  const float* x=xarr+xdir[3*q]+c
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[i*nc];
  rarr[rdir[2*q]+c]+=t;
}


__global__ void Ptensors1_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  extern __shared__ unsigned char _shared[]; 
  float* ix=reinterpret_cast<float*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xidir,xidir,q);
  const int nc=xdir[3*q+2];
  if(c>=nc) return;
  __syncthreads();

  const float* x=xarr+xdir[3*ix[0]]+c
  float t=0;
  for(int i=0; i<k; i++)
    t+=x[ix[i+1]*nc];
  rarr[rdir[2*q]+c]+=t;
}


__global__ void Ptensors1_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];

  const float* x=xarr+xdir[3*q]+c
  float* r=rarr+rdir[3*q]+c
  for(int i=0; i<k; i++)
    r[i*nc]+=x[i*nc];
}


__global__ void Ptensors1_reduce1_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  extern __shared__ unsigned char _shared[]; 
  float* ix=reinterpret_cast<float*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xidir,xidir,q);
  const int nc=xdir[3*q+2];
  if(c>=nc) return;
  __syncthreads();

  const float* x=xarr+xdir[3*ix[0]]+c
  float* r=rarr+rdir[3*q]+c
  for(int i=0; i<k; i++)
    r[i*nc]+=x[ix[i+1]*nc];
}


// ---- Broadcast --------------------------------------------------------------------------------------------


__global__ void Ptensors1_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];

  float* x=xarr+xdir[3*q]+c;
  const float t=rarr[rdir[2*q]+c]
  for(int i=0; i<k; i++)
    x[i*nc]+=t;
}


__global__ void Ptensors1_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xidir,xidir,q);
  const int nc=xdir[3*q+2];
  if(c>=nc) return;
  __syncthreads();

  float* x=xarr+xdir[3*ix[0]]+c
  const float t=rarr[rdir[2*q]+c]
  for(int i=0; i<k; i++)
    x[ix[i+1]*nc]+=t;
}


__global__ void Ptensors1_broadcast1_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=xdir[3*q+1];
  const int nc=xdir[3*q+2];

  float* x=xarr+xdir[3*q]+c;
  const float* r=rarr+rdir[3*q]+c
  for(int i=0; i<k; i++)
    x[i*nc]+=r[i*nc];
}


__global__ void Ptensors1_broadcast1_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir){
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  const int k=load_indices(ix,xidir,xidir,q);
  const int nc=xdir[3*q+2];
  if(c>=nc) return;
  __syncthreads();

  float* x=xarr+xdir[3*ix[0]]+c
  const float* r=rarr+rdir[3*q]+c
  for(int i=0; i<k; i++)
    x[ix[i+1]*nc]+=r[i*nc];
}


// -----------------------------------------------------------------------------------------------------------

namespace ptens{

  void Ptensors1_reduce0_cu(RtensorPack& R, const RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors1_reduce0_kernel<R.size(),n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg);
  }

  void Ptensors1_reduce0_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors1_reduce0_kernel<R.size(),std::max(n,32),32,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg);
  }

  void Ptensors1_reduce1_cu(RtensorPack& R, const RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors1_reduce1_kernel<R.size(),n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg);
  }

  void Ptensors1_reduce1_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors1_reduce1_kernel<R.size(),std::max(n,32),32,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg);
  }


  void Ptensors1_broadcast0_cu(RtensorPack& x, const RtensorPack& R, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors1_broadcast0_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors1_broadcast0_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast0_kernel<R.size(),std::max(n,32),32,stream>(x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors1_broadcast1_cu(RtensorPack& x, const RtensorPack& R, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors1_broadcast1_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors1_broadcast1_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast1_kernel<R.size(),std::max(n,32),32,stream>(x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg,R.arrg,r.dir.arrg);
  }

}


#endif 
