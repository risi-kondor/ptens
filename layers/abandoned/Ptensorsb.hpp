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

//#include "diff_class.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"

namespace ptens{


  template<typename TYPE>
  class Ptensorsb: public cnine::Ltensor<TYPE>{
  public:

    typedef cnine::Ltensor<TYPE> TENSOR;


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensorsb(const cnine::Gdims& _dims):
      TENSOR(_dims){}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensorsb(const Ptensorsb& x, const int _dev):
      TENSOR(x.copy(_dev)){}
    //atoms(x.atoms){
    //constk=x.constk;

    //Ptensorsb& to_device(const int _dev){
    //BASE::to_device(_dev);
    //return *this;
    //}



  };

}

#endif 
