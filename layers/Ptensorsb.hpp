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

#ifndef _ptens_Ptensorsb
#define _ptens_Ptensorsb

#include "diff_class.hpp"
#include "AtomsPack0.hpp"
#include "Ptensorsb.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"


namespace ptens{

#ifdef _WITH_ATEN
  template<typename TYPE>
  class ATview: public cnine::Ltensor<TYPE>{
  public:
    
    typedef cnine::Ltensor<TYPE> BASE;

    ATview(at::Tensor& x):
      BASE(BASE::view(x)){}

    ~ATview(){
      BASE::arr.blob->arr=nullptr;
    }

  };
#endif 


  template<typename TYPE, typename OBJ>
  class Ptensorsb: public cnine::Ltensor<TYPE>, public cnine::diff_class<OBJ>{
  public:

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    using BASE::BASE;
    using BASE::zeros_like;

    using cnine::diff_class<OBJ>::grad;
    using cnine::diff_class<OBJ>::get_grad;

    ~Ptensorsb(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void add_mprod_back0(const OBJ& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_scale_channels_back(const OBJ& g, const TENSOR& s){
      get_grad().add_scale_columns(g.get_grad(),s);
    }

    void add_linear_back0(const OBJ& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_ReLU_back(const OBJ& g, const OBJ& x, const float alpha){
      get_grad().add_ReLU_back(g.get_grad(),x,alpha);
    }

  };



  template<typename OBJ, typename TYPE>
  OBJ mprod(const OBJ& x, const cnine::Ltensor<TYPE>& y){
    return OBJ(x*y,x.atoms);
  }

  template<typename OBJ, typename TYPE>
  OBJ scale_channels(const OBJ& x, const cnine::Ltensor<TYPE>& s){
    return OBJ(x.scale_columns(s),x.atoms);
  }

  template<typename OBJ, typename TYPE>
  OBJ linear(const OBJ& x, const cnine::Ltensor<TYPE>& w, const cnine::Ltensor<TYPE>& b){
    OBJ R(x*w,x.atoms);
    R.add_broadcast(0,b);
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ ReLU(const OBJ& x, TYPE alpha){
    return OBJ(x.ReLU(alpha),x.atoms);
  }


  //  cnine::Ltensor<TYPE> mprod_back1(const OBJ& g, const OBJ& x){
  //return x.transp()*g.get_grad();
  //}

  //cnine::Ltensor<TYPE> linear_back1(const OBJ& g, const OBJ& x){
  //return x.transp()*g.get_grad();
  //}

  //cnine::Ltensor<TYPE> linear_back2(const OBJ& g){
  //return g.get_grad().sum(0);
  //}


}

#endif 
    //OBJ R=x.zeros_like();
    //R.add_scale_columns(x,s);
    //return R;
    //OBJ R=x.zeros_like();
    //R.add_ReLU(x,alpha);
    //return R;
    //    void _add_mprod(const OBJ& x, at::Tensor& M){
    //BASE::add_mprod(x,ATview<TYPE>(M));
    //}

    //void add_mprod_back0(const OBJ& g, at::Tensor& M){
    //get_grad().add_mprod(g.get_grad(),ATview<TYPE>(M).transp());
    //}

    //at::Tensor mprod_back1(const OBJ& x, const OBJ& r){
    //BASE R({x.dim(1),r.dim(1)},r.get_dev());
    //R.add_mprod(x.transp(),r.get_grad());
    //return R.torch();
    //}
