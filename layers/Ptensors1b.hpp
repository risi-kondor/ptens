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

#ifndef _ptens_Ptensors0b
#define _ptens_Ptensors0b

//#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Ptensors.hpp"
#include "Ptensor0.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"

namespace ptens{


  class Ptensors1b: public Ltensor<TYPE>, public cnine::diff_class<Ptensors1b>{
  public:

        ~Ptensors0(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1b(){}

    Ptensors1b(const int _nc, const int _dev=0):
      Ptensors(1,_nc,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Ptensors(_atoms, cnine::Gdims({_nc}), dummy, _dev){
      if(atoms.constk()>0) constk=atoms.constk();
    }

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const cnine::Tensor<int>& M, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Ptensors(AtomsPack(M), cnine::Gdims({_nc}), dummy, _dev){
      if(atoms.constk()>0) constk=atoms.constk();
    }

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Ptensors(AtomsPack(_n), cnine::Gdims({_nc}), dummy, _dev){
      constk=1;
    }


  public: // ----- Message passing ---------------------------------------------------------------------------

    
    transfer_from(const Ptensors0b<TYPE>& x){
      atoms.transfer_map(x.atoms)(*this,x);
    }

    transfer_from(const Ptensors1b<TYPE>& x){
      atoms.transfer_map(x.atoms)(*this,x);
    }

    transfer_from(const Ptensors2b<TYPE>& x){
      atoms.transfer_map(x.atoms)(*this,x);
    }


  };

#endif 

