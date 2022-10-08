#ifndef _Ptensors0_cu
#define _Ptensors0_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPack.hpp"
#include "Ptensors0.hpp"
//#include "Rtensor2_view.hpp"
//#include "Rtensor3_view.hpp"
//#include "Itensor1_view.hpp"
//#include "Itensor2_view.hpp"
//#include "CUDAhelpers.hpp"


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  rarr[rdir[2*i]+c]+=xarr[xdir[2*i]+c];
}


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  const int tix=xiarr[xidir[2*i]];
  rarr[rdir[2*i]+c]+=xarr[xdir[2*tix]+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  xarr[xdir[2*i]+c]+=rarr[rdir[2*i]+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  const int tix=xiarr[xidir[2*i]];
  xarr[xdir[2*i]+c]+=rarr[rdir[2*i]+c];
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{

  void Ptensors0_reduce0_cu(RtensorPack& R, const RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int N=R.size();
    Ptensors0_reduce0_kernel<N,n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg);
  }

  void Ptensors0_reduce0_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    int N=R.size();
    Ptensors0_reduce0_kernel<N,n,0,stream>(R.arrg,R.dir.arrg,x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg);
  }

  void Ptensors0_broadcast0_cu(RtensorPack& x, const RtensorPack& R, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors0_broadcast0_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,R.arrg,r.dir.arrg);
  }

  void Ptensors0_broadcast0_cu(RtensorPack& R, const RtensorPack& x, const AindexPack& list, const int offs, const cudaStream_t& stream){
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast0_kernel<R.size(),n,0,stream>(x.arrg+offs,x.dir.arrg,list.arrg,list.dir.arrg,R.arrg,r.dir.arrg);
  }

}

#endif 
