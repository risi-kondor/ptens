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

#ifndef _ptens_Ptensors
#define _ptens_Ptensors

#include "diff_class.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "AtomsPack.hpp"


namespace ptens{

  //using PtensTensor=cnine::Ltensor<TYPE>;


  template<typename TYPE>
  class Ptensors: public PtensTensor<TYPE>{
  public:

    typedef PtensTensor<TYPE> BASE;
    typedef PtensTensor<TYPE> TENSOR;
    using BASE::BASE;
    using BASE::zeros_like;
    using BASE::dim;

    AtomsPack atoms;
    int nc=0;

    virtual ~Ptensors(){}

    virtual Ptensors& get_grad(){CNINE_UNIMPL();return *this;} // dummy
    virtual const Ptensors& get_grad() const {CNINE_UNIMPL(); return *this;} // dummy


  public: // ---- Constructors -------------------------------------------------------------------------------------


    Ptensors(const AtomsPack& _atoms):
      atoms(_atoms){}

    Ptensors(const AtomsPack& _atoms, const BASE& x):
      BASE(x),
      atoms(_atoms){
      nc=TENSOR::dim(1);
    }

    Ptensors(const AtomsPack& _atoms, const cnine::Gdims& _dims, const int fcode, const int _dev):
      //BASE(BASE::vram_managed(ptens_session->managed_gmem,_dims,fcode,_dev)){
      BASE(_dims,fcode,_dev),
      atoms(_atoms){
      nc=TENSOR::dim(1);
    }


  public: // ---- Access ---------------------------------------------------------------------------------


    int size() const{
      return atoms.size();
    }

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    int get_nc() const{
      return nc;
      //return TENSOR::dim(1);
    }

    const AtomsPack& get_atoms() const{
      return atoms;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    //static OBJ cat(const vector<reference_wrapper<OBJ> >& list){
    //vector<shared_ptr<AtomsPackObjBase> > v; 
    //for(auto p:list)
    //v.push_back(p.get().atoms.obj);
    //return OBJ(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    //}

    void cat_channels_back0(const Ptensors& g){
      get_grad()+=g.get_grad().block(0,0,dim(0),dim(1));
    }

    void cat_channels_back1(const Ptensors& g){
      get_grad()+=g.get_grad().block(0,g.dim(1)-dim(1),dim(0),dim(1));
    }

    void add_mprod_back0(const Ptensors& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_scale_channels_back(const Ptensors& g, const TENSOR& s){
      get_grad().add_scale_columns(g.get_grad(),s);
    }

    void add_linear_back0(const Ptensors& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_ReLU_back(const Ptensors& g, const float alpha){
      get_grad().BASE::add_ReLU_back(g.get_grad(),*this,alpha);
    }

  };


  // this is used for BatchedPtensors as well 

  template<typename OBJ>
  OBJ cat_channels(const OBJ& x, const OBJ& y){
    PTENS_ASSRT(x.dim(0)==y.dim(0));
    //cnine::using_vram_manager vv(ptens_session->managed_gmem);
    OBJ R(typename OBJ::TENSOR(cnine::dims(x.dim(0),x.dim(1)+y.dim(1)),0,x.get_dev()),x.atoms);
    R.block(0,0,x.dim(0),x.dim(1))+=x;
    R.block(0,x.dim(1),x.dim(0),y.dim(1))+=y;
    return R;
  }

  /*
  template<typename OBJ, typename TYPE>
  OBJ scale_channels(const OBJ& x, const cnine::Ltensor<TYPE>& s){
    return OBJ(x.scale_columns(s),x.atoms);
  }

  template<typename OBJ, typename TYPE>
  OBJ mprod(const OBJ& x, const cnine::Ltensor<TYPE>& y){
    return OBJ(x*y,x.atoms);
  }

  template<typename OBJ, typename TYPE>
  OBJ linear(const OBJ& x, const cnine::Ltensor<TYPE>& w, const cnine::Ltensor<TYPE>& b){
    OBJ R(x*w,x.atoms);
    R.view2().add_broadcast0(b.view1());
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ ReLU(const OBJ& x, TYPE alpha){
    return OBJ(x.ReLU(alpha),x.atoms);
  }
  */

}

#endif 
