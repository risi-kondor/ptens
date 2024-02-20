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

#ifndef _ptens_BatchedPtensorsb
#define _ptens_BatchedPtensorsb

#include "diff_class.hpp"
#include "AtomsPack0.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"


namespace ptens{

  extern PtensSession ptens_session;


  template<typename TYPE>
  class BatchedPtensorsb: public cnine::Ltensor<TYPE>{
  public:

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    using BASE::BASE;
    using BASE::dims;
    using BASE::dev;
    using BASE::dim;
    using BASE::reset;


    virtual ~BatchedPtensorsb(){}


  public: // ---- Constructors -------------------------------------------------------------------------------------


    BatchedPtensorsb(const BASE& x):
      BASE(x){}

    //BatchedPtensorsb(const BatchedPtensorsb& x):
    //BASE(x){}

    BatchedPtensorsb(const cnine::Gdims& _dims, const int fcode, const int _dev){
      if(ptens_session.managed_gmem && _dev==1)
	reset(BASE(*ptens_session.managed_gmem,_dims,fcode,_dev));
      else
	reset(BASE(_dims,fcode,_dev));
    }

    BatchedPtensorsb copy(const BatchedPtensorsb& x){
      if(ptens_session.managed_gmem && x.get_dev()==1) return BASE::copy(*ptens_session.managed_gmem,x);
      else return BASE::copy(x); 
    }

    BatchedPtensorsb zeros_like() const{
      if(ptens_session.managed_gmem && dev==1) return BASE(*ptens_session.managed_gmem,dims,0,dev);
      else return BASE::zeros_like();
    }


  public: // ---- Access -------------------------------------------------------------------------------------

    virtual BatchedPtensorsb& get_grad(){CNINE_UNIMPL();return *this;} // dummy
    virtual const BatchedPtensorsb& get_grad() const {CNINE_UNIMPL(); return *this;} // dummy


  public: // ---- Operations ---------------------------------------------------------------------------------


    //static OBJ cat(const vector<reference_wrapper<OBJ> >& list){
    //vector<shared_ptr<AtomsPackObjBase> > v; 
    //for(auto p:list)
    //v.push_back(p.get().atoms.obj);
    //return OBJ(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    //}

    void cat_channels_back0(const BatchedPtensorsb& g){
      get_grad()+=g.get_grad().block(0,0,dim(0),dim(1));
    }

    void cat_channels_back1(const BatchedPtensorsb& g){
      get_grad()+=g.get_grad().block(0,g.dim(1)-dim(1),dim(0),dim(1));
    }

    void add_mprod_back0(const BatchedPtensorsb& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_scale_channels_back(const BatchedPtensorsb& g, const TENSOR& s){
      get_grad().add_scale_columns(g.get_grad(),s);
    }

    void add_linear_back0(const BatchedPtensorsb& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_ReLU_back(const BatchedPtensorsb& g, const float alpha){
      get_grad().BASE::add_ReLU_back(g.get_grad(),*this,alpha);
    }

  };


  /*
  template<typename OBJ, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, OBJ>::value, OBJ>::type>
  OBJ cat_channels(const OBJ& x, const OBJ& y){
    PTENS_ASSRT(x.dim(0)==y.dim(0));
    OBJ R({x.dim(0),x.dim(1)+y.dim(1)},0,x.get_dev());
    R.block(0,0,x.dim(0),x.dim(1))+=x;
    R.block(0,x.dim(1),x.dim(0),y.dim(1))+=y;
    return R;
  }

  template<typename OBJ, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, OBJ>::value, OBJ>::type, typename TYPE>
  OBJ scale_channels(const OBJ& x, const cnine::Ltensor<TYPE>& s){
    return OBJ(x.scale_columns(s),x.atoms);
  }

  template<typename OBJ, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, OBJ>::value, OBJ>::type, typename TYPE>
  OBJ mprod(const OBJ& x, const cnine::Ltensor<TYPE>& y){
    return OBJ(x*y,x.atoms);
  }

  template<typename OBJ, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, OBJ>::value, OBJ>::type, typename TYPE>
  OBJ linear(const OBJ& x, const cnine::Ltensor<TYPE>& w, const cnine::Ltensor<TYPE>& b){
    OBJ R(x*w,x.atoms);
    R.view2().add_broadcast0(b.view1());
    return R;
  }

  template<typename OBJ, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, OBJ>::value, OBJ>::type, typename TYPE>
  OBJ ReLU(const OBJ& x, TYPE alpha){
    return OBJ(x.ReLU(alpha),x.atoms);
  }
  */

}

#endif 
