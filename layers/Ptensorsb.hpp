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


  template<typename TYPE>
  class Ptensorsb: public cnine::Ltensor<TYPE>{ //, public cnine::diff_class<OBJ>{
  public:

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    using BASE::BASE;
    using BASE::zeros_like;
    using BASE::dim;

    virtual ~Ptensorsb(){}

    virtual Ptensorsb& get_grad(){return *this;} // dummy
    virtual const Ptensorsb& get_grad() const {return *this;} // dummy


  public: // ---- Operations ---------------------------------------------------------------------------------


    //static OBJ cat(const vector<reference_wrapper<OBJ> >& list){
    //vector<shared_ptr<AtomsPackObjBase> > v; 
    //for(auto p:list)
    //v.push_back(p.get().atoms.obj);
    //return OBJ(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    //}

    void cat_channels_back0(const Ptensorsb& g){
      get_grad()+=g.get_grad().block(0,0,dim(0),dim(1));
    }

    void cat_channels_back1(const Ptensorsb& g){
      get_grad()+=g.get_grad().block(0,g.dim(1)-dim(1),dim(0),dim(1));
    }

     void add_mprod_back0(const Ptensorsb& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_scale_channels_back(const Ptensorsb& g, const TENSOR& s){
      get_grad().add_scale_columns(g.get_grad(),s);
    }

    void add_linear_back0(const Ptensorsb& g, const TENSOR& M){
      get_grad().add_mprod(g.get_grad(),M.transp());
    }

    void add_ReLU_back(const Ptensorsb& g, const Ptensorsb& x, const float alpha){
      get_grad().BASE::add_ReLU_back(g.get_grad(),x,alpha);
    }

  };


  template<typename OBJ>
  OBJ cat_channels(const OBJ& x, const OBJ& y){
    //PTENS_ASSRT(x.atoms==y.atoms);
    PTENS_ASSRT(x.dim(0)==y.dim(0));
    OBJ R({x.dim(0),x.dim(1)+y.dim(1)},0,x.get_dev());
    R.block(0,0,x.dim(0),x.dim(1))+=x;
    R.block(0,x.dim(1),x.dim(0),y.dim(1))+=y;
    return R;
  }

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
    R.add_broadcast(0,b);
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ ReLU(const OBJ& x, TYPE alpha){
    return OBJ(x.ReLU(alpha),x.atoms);
  }

}

#endif 
