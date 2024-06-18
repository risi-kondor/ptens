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

  class EdgeLayer1;

  class EdgeLayer1: public cnine::Tensor<float>, public cnine::diff_class<NodeLayer>{
  public:

    typedef cnine::Tensor<float> BASE;

    const Ggraph G;
    AtomsPack atoms;
    int nc;

    ~NodeLayer(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    EdgeLayer1(const Ggraph& _G, const int _nc, const int _dev=0):
      EdgeLayer1(_G,_nc,cnine::fill_zero(),_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    EdgekLayer1(const Ggraph& _G, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      BASE({_G.nedges(),_nc}, dummy, _dev), G(_G),atoms(_G.edges()),nc(_nc){}


  public: // ----- Named Constructors ------------------------------------------------------------------------


  public: // ----- Copying -----------------------------------------------------------------------------------


    EdgeLayer1(const EdgeLayer1& x):
      BASE(x),
      cnine::diff_class<EdgeLayer1>(x),
      G(x.G), 
      atoms(x.atoms),
      nc(x.nc){
      PTENS_COPY_WARNING();
    }
	
    EdgeLayer1(EdgeLayer1&& x):
      BASE(std::move(x)),
      cnine::diff_class<EdgeLayer1>(std::move(x)),
      G(x.G), 
      atoms(x.atoms),
      nc(x.nc){
      PTENS_MOVE_WARNING();
    }
    
    EdgeLayer1& operator=(const EdgeLayer1& x)=delete;


  public: // ---- Spawning -----------------------------------------------------------------------------------


    // required for diff_class
    static EdgeLayer1* new_zeros_like(const EdgeLayer1& x){
      return new EdgeLayer1(x.G,x.atoms,BASE::zeros_like(x));
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    EdgeLayer1(const Ggraph& _G, const AtomsPack& _atoms, const cnine::TensorView<float>& x):
      BASE(x), G(_G), atoms(_atoms), nc(x.dim(2)){
      // check that it is regular!
      PTENS_ASSRT(x.ndims()==3);
      PTENS_ASSRT(x.dim(0)==_atoms.size());
      PTENS_ASSRT(x.dim(1)==2);
    }

//     #ifdef _WITH_ATEN
//     EdgeLayer1(const Ggraph& _G, const AtomsPack& _atoms, const at::Tensor& T):
//       NodeLayer(_G,_atoms,cnine::RtensorA::regular(T)){} // eliminate RtensorA
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
      return dim(2);
    }

    cnine::Rtensor2_view view_of(const int i) const{
      return cnine::Rtensor2_view(mem()+i*nc,2,nc,nc,1,dev);
    }

    //cnine::Rtensor2_view block_of(const int i, const int offs, const int n) const{
    //return cnine::Rtensor2_view(mem()+i*nc+offs,n,1,dev);
    //}


  public: // ----- Message passing from Ptensors -------------------------------------------------------------


    EdgeLayer1(const Ggraph& _G, const Ptensors0& x):
      EdgeLayer1(_G,2*x.get_nc(),cnine::fill_zero(),x.dev){
      emp_from(x);}

    void gather_back(Ptensors0& x){
      get_grad().emp_to(x.get_grad());}


    EdgeLayer1(const Ggraph& _G, const Ptensors1& x):
      EdgeLayer1(_G,5*x.get_nc(),cnine::fill_zero(),x.dev){
      emp_from(x);}

    void gather_back(Ptensors1& x){
      get_grad().emp_toB(x.get_grad());}


  public: // ----- Message passing from subgraph layers ------------------------------------------------------


    template<typename TLAYER>
    EdgeLayer1(const SubgraphLayer0<TLAYER>& x):
      EdgeLayer1(_G,2*x.get_nc(),cnine::fill_zero(),x.dev){
      emp_from(x);}

    template<typename TLAYER>
    void gather_back(SubgraphLayer0<TLAYER>& x){
      get_grad().emp_to(x.get_grad());}


    template<typename TLAYER>
    EdgeLayer1(const SubgraphLayer1<TLAYER>& x):
      EdgeLayer1(_G,5*x.get_nc(),cnine::fill_zero(),x.dev){
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





  };

}

#endif 
