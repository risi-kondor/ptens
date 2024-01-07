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

#ifndef _ptens_SubgraphLayerb
#define _ptens_SubgraphLayerb

namespace ptens{

  //template<typename OBJ>
  //OBJ sg_add(const OBJ& x, const OBJ& y){
  //return OBJ(x.G,x.S,x.add(y));
  //}

  template<typename OBJ>
    OBJ cat_channels_sg(const OBJ& x, const OBJ& y){
    //PTENS_ASSRT(x.atoms==y.atoms);
    PTENS_ASSRT(x.dim(0)==y.dim(0));
    OBJ R(x.G,x.S,cnine::Ltensor<float>({x.dim(0),x.dim(1)+y.dim(1)},0,x.get_dev()));
    R.block(0,0,x.dim(0),x.dim(1))+=x;
    R.block(0,x.dim(1),x.dim(0),y.dim(1))+=y;
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ scale_channels_sg(const OBJ& x, const cnine::Ltensor<TYPE>& s){
    return OBJ(x.G,x.S,x.scale_columns(s));
  }

  template<typename OBJ, typename TYPE>
  OBJ mprod_sg(const OBJ& x, const cnine::Ltensor<TYPE>& y){
    return OBJ(x.G,x.S,x*y);
  }

  template<typename OBJ, typename TYPE>
  OBJ linear_sg(const OBJ& x, const cnine::Ltensor<TYPE>& w, const cnine::Ltensor<TYPE>& b){
    OBJ R(x.G,x.S,x*w);
    R.add_broadcast(0,b);
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ ReLU_sg(const OBJ& x, TYPE alpha){
    return OBJ(x.G,x.S,x.ReLU(alpha));
  }


}


#endif 
