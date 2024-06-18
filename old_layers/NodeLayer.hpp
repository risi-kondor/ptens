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

#ifndef _ptens_NodeLayer
#define _ptens_NodeLayer

#include "diff_class.hpp"
#include "Tensor.hpp"
#include "Ggraph.hpp"


namespace ptens{

  class NodeLayer;

#ifdef _WITH_CUDA
  extern void NodeLayer_to_Ptensors0_cu(Ptensors0& r, const NodeLayer& x,  const cudaStream_t& stream);
  extern void NodeLayer_to_Ptensors1_cu(Ptensors1& r, const NodeLayer& x,  const cudaStream_t& stream);
  extern void NodeLayer_to_Ptensors1B_cu(Ptensors1& r, const NodeLayer& x,  const cudaStream_t& stream);
  extern void NodeLayer_from_Ptensors0_cu(NodeLayer& x, const Ptensors0& r, const cudaStream_t& stream);
  extern void NodeLayer_from_Ptensors1_cu(NodeLayer& x, const Ptensors1& r, const cudaStream_t& stream);
  extern void NodeLayer_from_Ptensors1B_cu(NodeLayer& x, const Ptensors1& r, const cudaStream_t& stream);
#endif 


  template<typename TLAYER> 
  class SubgraphLayer0;
  template<typename TLAYER> 
  class SubgraphLayer1;
  template<typename TLAYER> 
  class SubgraphLayer2;


  class NodeLayer: public cnine::Tensor<float>, public cnine::diff_class<NodeLayer>{
  public:

    typedef cnine::Tensor<float> BASE;

    const Ggraph G;
    int nc;

    ~NodeLayer(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    NodeLayer(const Ggraph& _G, const int _nc, const int _dev=0):
      NodeLayer(_G,_nc,cnine::fill_zero(),_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    NodeLayer(const Ggraph& _G, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      BASE({_G.getn(),_nc}, dummy, _dev), G(_G), nc(_nc){}


  public: // ----- Named Constructors ------------------------------------------------------------------------


  public: // ----- Copying -----------------------------------------------------------------------------------


    NodeLayer(const NodeLayer& x):
      BASE(x),
      cnine::diff_class<NodeLayer>(x),
      G(x.G), nc(x.nc){
      PTENS_COPY_WARNING();
    }
	
    NodeLayer(NodeLayer&& x):
      BASE(std::move(x)),
      cnine::diff_class<NodeLayer>(std::move(x)),
      G(x.G), nc(x.nc){
      PTENS_MOVE_WARNING();
    }
    
    NodeLayer& operator=(const NodeLayer& x)=delete;


  public: // ---- Spawning -----------------------------------------------------------------------------------


    // required for diff_class
    static NodeLayer* new_zeros_like(const NodeLayer& x){
      return new NodeLayer(x.G,BASE::zeros_like(x));
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    NodeLayer(const Ggraph& _G, const cnine::Tensor<float>& x):
      BASE(x), G(_G), nc(x.dim(1)){
      // check that it is regular!
      PTENS_ASSRT(x.ndims()==2);
      PTENS_ASSRT(_G.getn()==x.dim(0));
    }

//     #ifdef _WITH_ATEN
//     NodeLayer(const Ggraph& _G, const at::Tensor& T):
//       NodeLayer(_G,cnine::RtensorA::regular(T)){} // eliminate RtensorA
//     #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    NodeLayer(const NodeLayer& x, const int _dev):
      BASE(x,_dev),
      G(x.G), nc(x.nc){}

    NodeLayer& move_to_device(const int _dev){
      BASE::move_to_device(_dev);
      return *this;
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


    NodeLayer(const Ggraph& _G, const Ptensors0& x):
      BASE({_G.getn(),x.get_nc()},cnine::fill_zero(),x.dev),
      G(_G),
      nc(x.get_nc()){
      emp_from(x);}

    void gather_back(Ptensors0& x){
      get_grad().emp_to(x.get_grad());}


    NodeLayer(const Ggraph& _G, const Ptensors1& x):
      BASE({_G.getn(),2*x.get_nc()},cnine::fill_zero(),x.dev), 
      G(_G),
      nc(2*x.get_nc()){
      emp_from(x);}

    void gather_back(Ptensors1& x){
      get_grad().emp_toB(x.get_grad());}


  public: // ----- Message passing from subgraph layers ------------------------------------------------------


    template<typename TLAYER>
    NodeLayer(const SubgraphLayer0<TLAYER>& x):
      BASE({x.G.getn(),x.get_nc()},cnine::fill_zero(),x.dev),
      G(x.G),
      nc(x.get_nc()){
      emp_from(x);}

    template<typename TLAYER>
    void gather_back(SubgraphLayer0<TLAYER>& x){
      get_grad().emp_to(x.get_grad());}


    template<typename TLAYER>
    NodeLayer(const SubgraphLayer1<TLAYER>& x):
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
      cnine::flog timer("ptens::NodeLayer::emp_to(Ptensors0&)");

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

      GPUCODE(CUDA_STREAM(NodeLayer_to_Ptensors0_cu(x,*this,stream)));
    }

    void emp_to(Ptensors1& x) const{
      PTENS_ASSRT(x.get_nc()==2*get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayer::emp_to(Ptensors1&)");

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

      GPUCODE(CUDA_STREAM(NodeLayer_to_Ptensors1_cu(x,*this,stream)));
    }


    void emp_toB(Ptensors1& x) const{ 
      PTENS_ASSRT(x.get_nc()==get_nc()/2);
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayer::emp_toB(Ptensors1&)");

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

      GPUCODE(CUDA_STREAM(NodeLayer_to_Ptensors1B_cu(x,*this,stream)));
    }


    void emp_from(const Ptensors0& x){
      PTENS_ASSRT(x.get_nc()==get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayer::emp_from(Ptensors0&)");

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

      GPUCODE(CUDA_STREAM(NodeLayer_from_Ptensors0_cu(*this,x,stream)));
    }


    void emp_from(const Ptensors1& x){
      PTENS_ASSRT(get_nc()==2*x.get_nc());
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayer::emp_from(Ptensors1&)");

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

      GPUCODE(CUDA_STREAM(NodeLayer_from_Ptensors1_cu(*this,x,stream)));
    }


    void emp_fromB(const Ptensors1& x){
      PTENS_ASSRT(get_nc()==x.get_nc()/2);
      auto& xatoms=*x.atoms.obj;
      int N=x.size();
      int constk=x.constk;
      cnine::flog timer("ptens::NodeLayer::emp_fromB(Ptensors1&)");

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

      GPUCODE(CUDA_STREAM(NodeLayer_from_Ptensors1B_cu(*this,x,stream)));
    }


  public: // ----- I/O ---------------------------------------------------------------------------------------

    string classname() const{
      return "ptens::NodeLayer";
    }


  };

}

#endif 
