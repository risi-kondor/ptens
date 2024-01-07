/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _ptens_NodeLayerb
#define _ptens_NodeLayerb

#include "diff_class.hpp"
#include "Ltensor.hpp"
#include "Ggraph.hpp"


namespace ptens{

  class NodeLayer;

#ifdef _WITH_CUDA
  //  extern void NodeLayer_to_Ptensors0_cu(Ptensors0& r, const NodeLayer& x,  const cudaStream_t& stream);
  //extern void NodeLayer_to_Ptensors1_cu(Ptensors1& r, const NodeLayer& x,  const cudaStream_t& stream);
  //extern void NodeLayer_to_Ptensors1B_cu(Ptensors1& r, const NodeLayer& x,  const cudaStream_t& stream);
  //extern void NodeLayer_from_Ptensors0_cu(NodeLayer& x, const Ptensors0& r, const cudaStream_t& stream);
  //extern void NodeLayer_from_Ptensors1_cu(NodeLayer& x, const Ptensors1& r, const cudaStream_t& stream);
  //extern void NodeLayer_from_Ptensors1B_cu(NodeLayer& x, const Ptensors1& r, const cudaStream_t& stream);
#endif 


  template<typename TLAYER> class SubgraphLayer0b;
  template<typename TLAYER> class SubgraphLayer1;
  template<typename TLAYER> class SubgraphLayer2;


  class NodeLayerb: public cnine::Ltensor<float>, public cnine::diff_class<NodeLayerb>{
  public:

    typedef cnine::Tensor<float> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    const Ggraph G;
    int nc;

    ~NodeLayerb(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0b(const Ggraph& _G, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    NodeLayerb(const Ggraph& _G, const int _nc, const int fcode=0, const int _dev=0):
      BASE({_G.getn(),nc},0,_dev), G(_G), nc(_nc){}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    Ptensors0b copy() const{
      return Ptensors0b(G,TENSOR::copy());
    }

    Ptensors0b copy(const int _dev) const{
      return Ptensors0b(G,TENSOR::copy(_dev));
    }

    Ptensors0b zeros_like() const{
      return Ptensors0b(G,TENSOR::zeros_like());
    }

    Ptensors0b gaussian_like() const{
      return Ptensors0b(G,BASE::gaussian_like());
    }

    static Ptensors0b* new_zeros_like(const Ptensors0b& x){
      return new Ptensors0b(x.G,x.TENSOR::zeros_like());
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    int getn() const{
      return dim(0);
    }

    int get_nc() const{
      return dim(1);
    }

    cnine::Rtensor1_view view_of(const int i) const{
      return cnine::Rtensor1_view(mem()+i*nc,nc,1,dev);
    }

    cnine::Rtensor1_view block_of(const int i, const int offs, const int n) const{
      return cnine::Rtensor1_view(mem()+i*nc+offs,n,1,dev);
    }


  public: // ----- Message passing from Ptensors -------------------------------------------------------------


    NodeLayerb(const Ggraph& _G, const Ptensors0b& x):
      BASE({_G.getn(),x.get_nc()},cnine::fill_zero(),x.dev),
      G(_G),
      nc(x.get_nc()){
      emp_from(x);}

    void gather_back(Ptensors0& x){
      get_grad().emp_to(x.get_grad());}


    NodeLayerb(const Ggraph& _G, const Ptensors1& x):
      BASE({_G.getn(),2*x.get_nc()},cnine::fill_zero(),x.dev), 
      G(_G),
      nc(2*x.get_nc()){
      emp_from(x);}

    void gather_back(Ptensors1& x){
      get_grad().emp_toB(x.get_grad());}


  public: // ----- Message passing from subgraph layers ------------------------------------------------------


    template<typename TLAYER>
    NodeLayerb(const SubgraphLayer0<TLAYER>& x):
      BASE({x.G.getn(),x.get_nc()},cnine::fill_zero(),x.dev),
      G(x.G),
      nc(x.get_nc()){
      emp_from(x);}

    template<typename TLAYER>
    void gather_back(SubgraphLayer0<TLAYER>& x){
      get_grad().emp_to(x.get_grad());}


    template<typename TLAYER>
    NodeLayerb(const SubgraphLayer1<TLAYER>& x):
      BASE({x.G.getn(),2*x.get_nc()},cnine::fill_zero(),x.dev), 
      G(x.G),
      nc(x.get_nc()){
      emp_from(x);}

    template<typename TLAYER>
    void gather_back(SubgraphLayer1<TLAYER>& x){
      get_grad().emp_toB(x.get_grad());}



  public: // ----- Message passing ---------------------------------------------------------------------------


    void emp_to(Ptensors0& x) const{
      PTENS_ASSRT(x.get_nc()==get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayerb::emp_to(Ptensors0&)");

      if(dev==0){
	if(constk>0){
	  for(int i=0; i<N; i++){
	    auto u=x.constk_view_of(i);
	    for(int j=0; j<constk; j++)
	      u+=view_of(xatoms(i,j));
	  }
	}else{
	  for(int i=0; i<N; i++){
	    auto u=x.view_of(i);
	    for(int j=0; j<xatoms.size_of(i); j++)
	      u+=view_of(xatoms(i,j));
	  }
	}
      }

      GPUCODE(CUDA_STREAM(NodeLayerb_to_Ptensors0_cu(x,*this,stream)));
    }

    void emp_to(Ptensors1& x) const{
      PTENS_ASSRT(x.get_nc()==2*get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayerb::emp_to(Ptensors1&)");

      if(dev==0){
	if(constk>0){
	  for(int i=0; i<N; i++){
	    auto u=x.constk_view_of(i,0,nc);
	    for(int j=0; j<constk; j++)
	      u+=repeat0(view_of(xatoms(i,j)),constk);
	    auto v=x.constk_view_of(i,nc,nc);
	    for(int j=0; j<constk; j++)
	      v.slice0(j)+=view_of(xatoms(i,j));
	  }
	}else{
	  for(int i=0; i<N; i++){
	    int k=xatoms.size_of(i);
	    auto u=x.view_of(i,0,nc);
	    for(int j=0; j<k; j++)
	      u+=repeat0(view_of(xatoms(i,j)),k);
	    auto v=x.view_of(i,nc,nc);
	    for(int j=0; j<k; j++)
	      v.slice0(j)+=view_of(xatoms(i,j));
	  }
	}
      }

      GPUCODE(CUDA_STREAM(NodeLayerb_to_Ptensors1_cu(x,*this,stream)));
    }


    void emp_toB(Ptensors1& x) const{ 
      PTENS_ASSRT(x.get_nc()==get_nc()/2);
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayerb::emp_toB(Ptensors1&)");

      if(dev==0){
	if(constk>0){
	  for(int i=0; i<N; i++){
	    auto u=x.constk_view_of(i);
	    for(int j=0; j<constk; j++)
	      u+=repeat0(block_of(xatoms(i,j),0,nc/2),constk);
	    for(int j=0; j<constk; j++)
	      u.slice0(j)+=block_of(xatoms(i,j),nc/2,nc/2);
	  }
	}else{
	  for(int i=0; i<N; i++){
	    int k=xatoms.size_of(i);
	    auto u=x.view_of(i);
	    for(int j=0; j<k; j++)
	      u+=repeat0(block_of(xatoms(i,j),0,nc/2),k);
	    for(int j=0; j<k; j++)
	      u.slice0(j)+=block_of(xatoms(i,j),nc/2,nc/2);
	  }
	}
      }

      GPUCODE(CUDA_STREAM(NodeLayerb_to_Ptensors1B_cu(x,*this,stream)));
    }


    void emp_from(const Ptensors0& x){
      PTENS_ASSRT(x.get_nc()==get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayerb::emp_from(Ptensors0&)");

      if(dev==0){
	if(constk>0){
	  for(int i=0; i<N; i++){
	    auto u=x.constk_view_of(i);
	    for(int j=0; j<constk; j++)
	      view_of(xatoms(i,j))+=u;
	  }
	}else{
	  for(int i=0; i<N; i++){
	    auto u=x.view_of(i);
	    for(int j=0; j<xatoms.size_of(i); j++)
	      view_of(xatoms(i,j))+=u;
	  }
	}
      }

      GPUCODE(CUDA_STREAM(NodeLayerb_from_Ptensors0_cu(*this,x,stream)));
    }


    void emp_from(const Ptensors1& x){
      PTENS_ASSRT(get_nc()==2*x.get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayerb::emp_from(Ptensors1&)");

      if(dev==0){
	if(constk>0){
	  Tensor<float> t({nc/2},cnine::fill_raw(),dev);
	  for(int i=0; i<N; i++){
	    t.set_zero();
	    x.constk_view_of(i).sum0_into(t);
	    auto xview_i=x.constk_view_of(i);
	    for(int j=0; j<constk; j++){
	      block_of(xatoms(i,j),0,nc/2)+=t.view1();
	      block_of(xatoms(i,j),nc/2,nc/2)+=xview_i.slice0(j);
	    }
	  }
	}else{
	  Tensor<float> t({nc/2},cnine::fill_raw(),dev);
	  for(int i=0; i<N; i++){
	    int k=xatoms.size_of(i);
	    t.set_zero();
	    x.view_of(i).sum0_into(t);
	    auto xview_i=x.view_of(i);
	    for(int j=0; j<k; j++){
	      block_of(xatoms(i,j),0,nc/2)+=t.view1();
	      block_of(xatoms(i,j),nc/2,nc/2)+=xview_i.slice0(j);
	    }
	  }
	}
      }

      GPUCODE(CUDA_STREAM(NodeLayerb_from_Ptensors1_cu(*this,x,stream)));
    }


    void emp_fromB(const Ptensors1& x){
      PTENS_ASSRT(get_nc()==x.get_nc()/2);
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayerb::emp_fromB(Ptensors1&)");

      if(dev==0){
	if(constk>0){
	  Tensor<float> t({nc},cnine::fill_raw(),dev);
	  for(int i=0; i<N; i++){
	    t.set_zero();
	    x.constk_view_of(i,0,nc).sum0_into(t);
	    auto xview_i=x.constk_view_of(i,nc,nc);
	    for(int j=0; j<constk; j++){
	      view_of(xatoms(i,j))+=t.view1();
	      view_of(xatoms(i,j))+=xview_i.slice0(j);
	    }
	  }
	}else{
	  Tensor<float> t({nc},cnine::fill_raw(),dev);
	  for(int i=0; i<N; i++){
	    int k=xatoms.size_of(i);
	    t.set_zero();
	    x.view_of(i,0,nc).sum0_into(t);
	    auto xview_i=x.view_of(i,nc,nc);
	    for(int j=0; j<k; j++){
	      view_of(xatoms(i,j))+=t.view1();
	      view_of(xatoms(i,j))+=xview_i.slice0(j);
	    }
	  }
	}
      }

      GPUCODE(CUDA_STREAM(NodeLayerb_from_Ptensors1B_cu(*this,x,stream)));
    }


  public: // ----- I/O ---------------------------------------------------------------------------------------

    string classname() const{
      return "ptens::NodeLayerb";
    }


  };

}

#endif 
