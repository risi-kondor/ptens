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

#ifndef _ptens_SubgraphLayer
#define _ptens_SubgraphLayer


namespace ptens{


  template<typename OBJ>
    OBJ cat_channels_sg(const OBJ& x, const OBJ& y){
    PTENS_ASSRT(x.dim(0)==y.dim(0));
    OBJ R(x.G,x.S,x.atoms,cnine::Ltensor<float>({x.dim(0),x.dim(1)+y.dim(1)},0,x.get_dev()));
    R.block(0,0,x.dim(0),x.dim(1))+=x;
    R.block(0,x.dim(1),x.dim(0),y.dim(1))+=y;
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ scale_channels_sg(const OBJ& x, const cnine::Ltensor<TYPE>& s){
    return OBJ(x.G,x.S,x.atoms,x.scale_columns(s));
  }

  template<typename OBJ, typename TYPE>
  OBJ mprod_sg(const OBJ& x, const cnine::Ltensor<TYPE>& y){
    return OBJ(x.G,x.S,x.atoms,x*y);
  }

  template<typename OBJ, typename TYPE>
  OBJ linear_sg(const OBJ& x, const cnine::Ltensor<TYPE>& w, const cnine::Ltensor<TYPE>& b){
    OBJ R(x.G,x.S,x.atoms,x*w);
    R.view2().add_broadcast0(b.view1());
    return R;
  }

  template<typename OBJ, typename TYPE>
  OBJ ReLU_sg(const OBJ& x, TYPE alpha){
    return OBJ(x.G,x.S,x.atoms,x.ReLU(alpha));
  }

}


#endif 
