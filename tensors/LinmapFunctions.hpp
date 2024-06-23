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
#ifndef _ptens_LinmapFunctions
#define _ptens_LinmapFunctions

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"


namespace ptens{
  
  template<typename TYPE>
  inline Ptensor0<TYPE> linmaps0(const Ptensor0<TYPE>& x){
    Ptensor0<TYPE> R=Ptensor0<TYPE>::zero(x.atoms,x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensor0<TYPE> linmaps0(const Ptensor1<TYPE>& x){
    Ptensor0<TYPE> R=Ptensor0<TYPE>::zero(x.atoms,x.nc,x.dev);
    x.add_linmaps_to(R);
    return R;
  }

  template<typename TYPE>
  inline Ptensor0<TYPE> linmaps0(const Ptensor2<TYPE>& x){
    Ptensor0<TYPE> R=Ptensor0<TYPE>::zero(x.atoms,2*x.nc,x.dev);
    x.add_linmaps_to(R);
    return R;
  }


  template<typename TYPE>
  inline Ptensor1<TYPE> linmaps1(const Ptensor0<TYPE>& x){
    Ptensor1<TYPE> R=Ptensor1<TYPE>::zero(x.atoms,x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensor1<TYPE> linmaps1(const Ptensor1<TYPE>& x){
    Ptensor1<TYPE> R=Ptensor1<TYPE>::zero(x.atoms,2*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensor1<TYPE> linmaps1(const Ptensor2<TYPE>& x){
    Ptensor1<TYPE> R=Ptensor1<TYPE>::zero(x.atoms,5*x.nc,x.dev);
    x.add_linmaps_to(R);
    return R;
  }


  template<typename TYPE>
  inline Ptensor2<TYPE> linmaps2(const Ptensor0<TYPE>& x){
    Ptensor2<TYPE> R=Ptensor2<TYPE>::zero(x.atoms,2*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensor2<TYPE> linmaps2(const Ptensor1<TYPE>& x){
    Ptensor2<TYPE> R=Ptensor2<TYPE>::zero(x.atoms,5*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensor2<TYPE> linmaps2(const Ptensor2<TYPE>& x){
    Ptensor2<TYPE> R=Ptensor2<TYPE>::zero(x.atoms,15*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

}


#endif 

