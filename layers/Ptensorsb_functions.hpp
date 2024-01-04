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

#ifndef _ptens_Ptensorsb_functions
#define _ptens_Ptensorsb_functions

#include "Ptensorsb0.hpp"
#include "Ptensorsb1.hpp"
#include "Ptensorsb2.hpp"

namespace ptens{

  template<typename TYPE>
  inline linmaps0(const Ptensorsb<TYPE>& x){
    Ptensorsb0<TYPE> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline linmaps1(const Ptensorsb<TYPE>& x){
    Ptensorsb1<float> R(x.atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline linmaps2(const Ptensorsb<TYPE>& x){
    Ptensorsb2<float> R(x.atoms,x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

};

#endif 
