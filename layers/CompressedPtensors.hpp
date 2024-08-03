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

#ifndef _ptens_CompressedPtensors
#define _ptens_CompressedPtensors

#include "diff_class.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "SpectralAtomsPack.hpp"


namespace ptens{


  template<typename TYPE>
  class CompressedPtensors: public cnine::Ltensor<TYPE>{
  public:

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    using BASE::BASE;
    using BASE::zeros_like;
    using BASE::dim;

    SpectralAtomsPack atoms;
    int nc=0;

    virtual ~CompressedPtensors(){}

    //virtual CompressedPtensors& get_grad(){CNINE_UNIMPL();return *this;} // dummy
    //virtual const CompressedPtensors& get_grad() const {CNINE_UNIMPL(); return *this;} // dummy


  public: // ---- Constructors -------------------------------------------------------------------------------------


    CompressedPtensors(const SpectralAtomsPack& _atoms):
      atoms(_atoms){}

    CompressedPtensors(const SpectralAtomsPack& _atoms, const BASE& x):
      BASE(x),
      atoms(_atoms){
      nc=TENSOR::dim(1);
    }

    CompressedPtensors(const SpectralAtomsPack& _atoms, const cnine::Gdims& _dims, const int fcode, const int _dev):
      BASE(_dims,fcode,_dev),
      atoms(_atoms){
      nc=TENSOR::dim(1);
    }


  public: // ---- Access ---------------------------------------------------------------------------------


    int size() const{
      return atoms.size();
    }

    //Atoms atoms_of(const int i) const{
    //return atoms(i);
    //}
    
    int get_nc() const{
      return nc;
      //return TENSOR::dim(1);
    }

    const SpectralAtomsPack& get_atoms() const{
      return atoms;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------



  };
