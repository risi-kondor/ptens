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

#include "NodeLayer.hpp"


__global__ void NodeLayer_to_Ptensors0_kernel(float* rarr, const int* rdir, const float* xarr, 
  const int* atomsarr, const int* atomsdir){

  const int b=blockIdx.x;
  const int c=threadIdx.x;

  int* atoms=atomsarr+atomsdir[2*b];
  int natoms=atomsdir[2*b+1];
  
  float t=0;
  for(int a=0; a<natoms; a++)
    t+=xarr[atoms[a]*nc+c];
  rarr[rdir[2*i]+c]+=t;
}


__global__ void NodeLayer_to_Ptensors1_kernel(float* rarr, const int* rdir, const float* xarr, 
  const int* atomsarr, const int* atomsdir){

  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int nc=blockDim.x;

  int* atoms=atomsarr+atomsdir[2*b];
  int natoms=atomsdir[2*b+1];
  
  float t=0;
  for(int i=0; i<natoms; i++)
    t+=xarr[atoms[i]*nc+c];
  for(int i=0; i<natoms; i++)
    rarr[rdir[2*i]+2*i*nc+c]+=t;

  for(int i=0; i<natoms; i++)
    rarr[rdir[2*i]+2*i*nc+nc+c]+=xarr[atoms[i]*nc+c];
}


// shrinking version, nc is half the number of channels in x
__global__ void NodeLayer_to_Ptensors1_kernelB(float* rarr, const int* rdir, const float* xarr, 
  const int* atomsarr, const int* atomsdir){

  const int b=blockIdx.x;
  const int c=threadIdx.x;
  const int nc=blockDim.x;

  int* atoms=atomsarr+atomsdir[2*b];
  int natoms=atomsdir[2*b+1];
  
  float t=0;
  for(int i=0; i<natoms; i++)
    t+=xarr[atoms[i]*2*nc+c];
  for(int i=0; i<natoms; i++)
    rarr[rdir[2*i]+i*nc+c]+=t+
      xarr[atoms[i]*2*nc+nc+c];
}


__global__ void NodeLayer_from_Ptensors0_kernel(float* rarr, const float* xarr, const int* xdir, 
  const int* bmap){

  const int b=blockIdx.x;
  const int c=threadIdx.x;

  const int offs=bmap[3*b];
  const int n=bmap[3*b+1];
  const int target=bmap[3*b+2];

  float t=0;
  for(int i=0; i<n; i++)
    t+=xarr[xdir[2*bmap[offs+2*i]]+c];
  rarr[target*nc+c]+=t;
}


__global__ void NodeLayer_from_Ptensors1_kernel(float* rarr, const float* xarr, const int* xdir, 
  const int* atomsarr, const int* atomsdir, const int* bmap){

  const int b=blockIdx.x;
  const int c=threadIdx.x;

  const int offs=bmap[3*b];
  const int n=bmap[3*b+1];
  const int target=bmap[3*b+2];

  float tsum=0;
  for(int i=0; i<n; i++){
    int src=bmap[offs+2*i];
    int offs=xdir[2*src];
    int aoffs=atomsdir[2*src];
    int k=atomsdir[2*src+1];

    for(int j=0; j<k; i++){
      tsum+=xarr[offs+nc*j+c];
      if(atomsarr[aoffs+j]==target)
	rarr[target*2*nc+nc+c]+=xarr[offs+nc*j+c]
    }
  }
  rarr[target*2*nc+c]+=tsum;
}


// shrinking version, nc is half the number of channels in x
__global__ void NodeLayer_from_Ptensors1B_kernel(float* rarr, const float* xarr, const int* xdir, 
  const int* atomsarr, const int* atomsdir, const int* bmap){

  const int b=blockIdx.x;
  const int c=threadIdx.x;

  const int offs=bmap[3*b];
  const int n=bmap[3*b+1];
  const int target=bmap[3*b+2];

  float tsum=0;
  for(int i=0; i<n; i++){
    int src=bmap[offs+2*i];
    int offs=xdir[2*src];
    int aoffs=atomsdir[2*src];
    int k=atomsdir[2*src+1];

    for(int j=0; j<k; i++){
      tsum+=xarr[offs+2*nc*j+c];
      if(atomsarr[aoffs+j]==target)
	rarr[target*nc+c]+=xarr[offs+2*nc*j+nc+c]
    }
  }
  rarr[target*nc+c]+=tsum;
}






namespace ptens{


  void NodeLayer_to_Ptensors0_cu(Ptensors0& r, const NodeLayer& x,  const cudaStream_t& stream){
    int dev=x.dev;
    PTENS_ASSRT(r.dev==dev);
    auto atoms=r.atoms.obj->gpack;

    NodeLayer_to_Ptensors0_kernel<<<r.size(),x.nc,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arr,atoms.arrg,atoms.dir.arrg);
  }

  void NodeLayer_to_Ptensors1_cu(Ptensors1& r, const NodeLayer& x,  const cudaStream_t& stream){
    int dev=x.dev;
    PTENS_ASSRT(r.dev==dev);
    auto atoms=r.atoms.obj->gpack;

    NodeLayer_to_Ptensors1_kernel<<<r.size(),x.nc,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arr,atoms.arrg,atoms.dir.arrg);
  }

  void NodeLayer_to_Ptensors1B_cu(Ptensors1& r, const NodeLayer& x,  const cudaStream_t& stream){
    int dev=x.dev;
    PTENS_ASSRT(r.dev==dev);
    auto atoms=r.atoms.obj->gpack;

    NodeLayer_to_Ptensors1B_kernelB<<<r.size(),x.nc/2,0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arr,atoms.arrg,atoms.dir.arrg);
  }


  void NodeLayer_from_Ptensors0_cu(NodeLayer& r, const Ptensors0& x,  const cudaStream_t& stream){
    int dev=x.dev;
    PTENS_ASSRT(r.dev==dev);
    auto atoms=x.atoms.obj->gpack;

    NodeLayer_from_Ptensors0_kernel<<<r.size(),x.nc,0,stream>>>
      (r.arr,x.arrg,x.dir.garr(dev),x.atoms.obj->to_nodes_map->get_barr(1));
  }

  void NodeLayer_from_Ptensors1_cu(NodeLayer& r, const Ptensors1& x,  const cudaStream_t& stream){
    int dev=x.dev;
    PTENS_ASSRT(r.dev==dev);
    auto atoms=x.atoms.obj->gpack;

    NodeLayer_from_Ptensors1_kernel<<<r.size(),x.nc,0,stream>>>
      (r.arr, x.arrg, x.dir.garr(dev), atoms.arrg, atoms.dir.arrg, x.atoms.obj->to_nodes_map->get_barr(1));
  }

  void NodeLayer_from_Ptensors1B_cu(NodeLayer& r, const Ptensors1& x,  const cudaStream_t& stream){
    int dev=x.dev;
    PTENS_ASSRT(r.dev==dev);
    auto atoms=x.atoms.obj->gpack;

    NodeLayer_from_Ptensors1B_kernel<<<r.size(),x.nc,0,stream>>>
      (r.arr, x.arrg, x.dir.garr(dev), atoms.arrg, atoms.dir.arrg, x.atoms.obj->to_nodes_map->get_barr(1));
  }


}

#endif 
