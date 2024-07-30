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

#include <cuda.h>
#include <cuda_runtime.h>

#include "Ptens_base.hpp"
#include "Ltensor.hpp"
#include "AindexPackB.hpp"


typedef cnine::Ltensor<float> TENSOR;
typedef cnine::Ltensor<int> ITENSOR;



// ---- Reduce -----------------------------------------------------------------------------------------------


__global__ void Ptensors2_reduce0_kernel(float* rarr, int rs, const float* xarr, int xs, const int* maparr, int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  const int k=ix[1];
  const int m=ix[3];
  __syncthreads();
  if(c>=n) return;
  const float* x=xarr+ix[2]*xs+c;

  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(ix[i+4]*m+ix[j+4])*xs];
  rarr[ix[0]*rs+c]+=t;

  t=0;
  for(int i=0; i<k; i++)
    t+=x[ix[i+4]*(m+1)*xs];
  rarr[ix[0]*rs+n+c]+=t;
}


__global__ void Ptensors2_reduce0_shrink_kernel(float* rarr, int rs, const float* xarr, int xs, const int* maparr, int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);
  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  const int k=ix[1];
  const int m=ix[3];
  __syncthreads();
  if(c>=n) return;
  const float* x=xarr+ix[2]*xs+c;

  float t=0;
  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      t+=x[(ix[i+4]*m+ix[j+4])*xs];
  rarr[ix[0]*rs+c]+=t;

  t=0;
  x+=n;
  for(int i=0; i<k; i++)
    t+=x[ix[i+4]*(m+1)*xs];
  rarr[ix[0]*rs+c]+=t;
}


__global__ void Ptensors2_reduce1_kernel(float* rarr, int rs, const float* xarr, const int xs, 
  const int* maparr, const int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  __syncthreads();
  if(c>=n) return;

  const int k=ix[1];
  const int m=ix[3];
  const float* x=xarr+ix[2]*xs+c;
  float* r=rarr+ix[0]*rs+c;

  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[j+4]*m+ix[i+4])*xs];
    r[i*rs]+=t;
  }

  r+=n;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[i+4]*m+ix[j+4])*xs];
    r[i*rs]+=t;
  }

  r+=n;
  for(int i=0; i<k; i++)
    r[i*rs]+=x[ix[i+4]*(m+1)*xs];
}


__global__ void Ptensors2_reduce1_shrink_kernel(float* rarr, int rs, const float* xarr, const int xs, 
  const int* maparr, const int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  __syncthreads();
  if(c>=n) return;

  const int k=ix[1];
  const int m=ix[3];
  const float* x=xarr+ix[2]*xs+c;
  float* r=rarr+ix[0]*rs+c;

  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[j+4]*m+ix[i+4])*xs];
    r[i*rs]+=t;
  }

  x+=n;
  for(int i=0; i<k; i++){
    float t=0;
    for(int j=0; j<k; j++)
      t+=x[(ix[i+4]*m+ix[j+4])*xs];
    r[i*rs]+=t;
  }

  x+=n;
  for(int i=0; i<k; i++)
    r[i*rs]+=x[ix[i+4]*(m+1)*xs];
}


__global__ void Ptensors2_reduce2_kernel(float* rarr, int rs, const float* xarr, const int xs, 
  const int* maparr, const int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  __syncthreads();
  if(c>=n) return;

  const int k=ix[1];
  const int m=ix[3];
  const float* x=xarr+ix[2]*xs+c;
  float* r=rarr+ix[0]*rs+c;

  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*rs]+=x[(ix[i+4]*m+ix[j+4])*xs];
}


__global__ void Ptensors2_reduce2_shrink_kernel(float* rarr, int rs, const float* xarr, const int xs, 
  const int* maparr, const int maps, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int q=blockIdx.x;
  const int c=threadIdx.x;
  if(c<maps) ix[c]=maparr[q*maps+c];
  __syncthreads();
  if(c>=n) return;

  const int k=ix[1];
  const int m=ix[3];
  const float* x=xarr+ix[2]*xs+c;
  float* r=rarr+ix[0]*rs+c;

  for(int i=0; i<k; i++)
    for(int j=0; j<k; j++)
      r[(i*k+j)*rs]+=x[(ix[i+4]*m+ix[j+4])*xs]+x[(ix[j+4]*m+ix[i+4])*xs];
}


// ---- Broadcast --------------------------------------------------------------------------------------------


__global__ void Ptensors2_broadcast0_kernel(float* rarr, const int rs, 
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
    __syncthreads();
    if(c>=n) continue;
    const int k=ix[1];
    const int m=ix[3];

    float t=xarr[ix[0]*xs+c];
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	r[(ix[i+4]+ix[j+4]*m)*rs]+=t;
    for(int i=0; i<k; i++)
      r[ix[i+4]*(m+1)*rs+n]+=t;
  }
}


__global__ void Ptensors2_broadcast0_shrink_kernel(float* rarr, const int rs, 
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
    __syncthreads();
    if(c>=n) continue;
    const int k=ix[1];
    const int m=ix[3];

    float t=xarr[ix[0]*xs+c];
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++)
	r[(ix[i+4]+ix[j+4]*m)*rs]+=t;

    t=xarr[ix[0]*xs+n+c];
    for(int i=0; i<k; i++)
      r[ix[i+4]*(m+1)*rs]+=t;
  }
}


__global__ void Ptensors2_broadcast1_kernel(float* rarr, const int rs, 
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
    __syncthreads();
    if(c>=n) continue;
    const int k=ix[1];
    const int m=ix[3];

    const float* x=xarr+ix[0]*xs+c;
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++){
      float t=x[i*xs];
      for(int j=0; j<k; j++){
	r[(ix[i+4]+ix[j+4]*m)*rs]+=t;
	r[(ix[j+4]+ix[i+4]*m)*rs+n]+=t;
      }
      r[ix[i+4]*(m+1)*rs+2*n]+=t;
    }
  }
}


__global__ void Ptensors2_broadcast1_shrink_kernel(float* rarr, const int rs, 
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
    __syncthreads();
    if(c>=n) continue;
    const int k=ix[1];
    const int m=ix[3];

    const float* x=xarr+ix[0]*xs+c;
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++){
      float t=x[i*xs];
      for(int j=0; j<k; j++)
	r[(ix[i+4]+ix[j+4]*m)*rs]+=t;
      t=x[i*xs+n];
      for(int j=0; j<k; j++)
	r[(ix[j+4]+ix[i+4]*m)*rs]+=t;
      r[ix[i+4]*(m+1)*rs]+=x[i*xs+2*n];
    }
  }
}


__global__ void Ptensors2_broadcast2_kernel(float* rarr, const int rs, 
  const float* xarr, const int xs, const int* maparr, const int maps, const int* bmap, const int n){
  extern __shared__ unsigned char _shared[]; 
  int* ix=reinterpret_cast<int*>(_shared);

  const int b=blockIdx.x;
  assert(b<bmap[0]);
  const int c=threadIdx.x;
  const int boffs=bmap[b+1];
  const int N=bmap[b+2]-bmap[b+1]-1;

  for(int s=0; s<N; s++){
    const int row=bmap[boffs+s+1];
    if(c<maps) ix[c]=maparr[row*maps+c];
    __syncthreads();
    if(c>=n) continue;
    const int k=ix[1];
    const int m=ix[3];

    const float* x=xarr+ix[0]*xs+c;
    float* r=rarr+ix[2]*rs+c;
    for(int i=0; i<k; i++)
      for(int j=0; j<k; j++){
	float t=x[(i*k+j)*xs];
	r[(ix[i+4]*m+ix[j+4])*rs]+=t;
	r[(ix[j+4]*m+ix[i+4])*rs+n]+=t;
      }
  }
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{


  void Ptensors2_reduce0_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    if(map.dim(0)==0) return;

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    Ptensors2_reduce0_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors2_reduce0_shrink_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    if(map.dim(0)==0) return;

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    Ptensors2_reduce0_shrink_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors2_reduce1_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)+1),32);
    Ptensors2_reduce1_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors2_reduce1_shrink_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)+1),32);
    Ptensors2_reduce1_shrink_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors2_reduce2_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)+1),32);
    Ptensors2_reduce2_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }

  void Ptensors2_reduce2_shrink_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);

    const int nthrd=cnine::roundup(std::max(n,map.dim(1)+1),32);
    Ptensors2_reduce2_shrink_kernel<<<map.dim(0),nthrd,map.dim(1)*4,stream>>>
      (r.get_arr(),r.stride(0),x.get_arr()+offs,x.stride(0),map.on_device(dev).get_arr(),map.stride(0),n);
  }


  void Ptensors2_broadcast0_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors2_broadcast0_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

  void Ptensors2_broadcast0_shrink_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors2_broadcast0_shrink_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

  void Ptensors2_broadcast1_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors2_broadcast1_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

  void Ptensors2_broadcast1_shrink_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors2_broadcast1_shrink_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

  void Ptensors2_broadcast2_cu(const TENSOR& r, const TENSOR& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(x.dev==dev);
    int n=x.dim(1);

    int nthrd=cnine::roundup(std::max(n,map.dim(1)),32);
    if(map.n_gather_lists==0) return;
    Ptensors2_broadcast2_kernel<<<map.n_gather_lists,nthrd,map.dim(1)*4,stream>>> 
      (r.get_arr()+offs,r.stride(0),x.get_arr(),x.stride(0),map.on_device(dev).get_arr(),map.stride(0),
	map.gmap_on_device(dev).get_arr(),n);
  }

}

