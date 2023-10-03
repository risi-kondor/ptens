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

#ifndef _ptens_Ptensors
#define _ptens_Ptensors

#include "Ptens_base.hpp"

//#include "Cgraph.hpp"
#include "RtensorPackB.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor0.hpp"
#include "loose_ptr.hpp"
#include "diff_class.hpp"

#include "PtensLoggedTimer.hpp"


namespace ptens{

  class Ptensors: public cnine::RtensorPackB{
  public:

    AtomsPack atoms;


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors(){}

    Ptensors(const int k, const int _nc, const int _dev=0):
      RtensorPackB(k,_nc,_dev){}

    //Ptensors(const AtomsPack& _atoms, const Gdims& _dims, const int _dev=0):
    //RtensorPackB(_atoms.size(),_dims,_dev), atoms(_atoms){}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ptensors(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
    //RtensorPackB(_atoms.size(), cnine::Gdims({_nc}), dummy, _dev), atoms(_atoms){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors(const AtomsPack& _atoms, const cnine::Gdims& _dims, const FILLTYPE& dummy, const int _dev=0):
      RtensorPackB(_atoms.size(), _dims, dummy, _dev), atoms(_atoms){}


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors(const Ptensors& x):
      RtensorPackB(x),
      atoms(x.atoms){}

    Ptensors(Ptensors&& x):
      RtensorPackB(std::move(x)),
      atoms(std::move(x.atoms)){}


  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors(const RtensorPackB& x, const AtomsPack& _atoms):
      RtensorPackB(x), 
      atoms(_atoms){}

    Ptensors(RtensorPackB&& x, const AtomsPack& _atoms):
      RtensorPackB(std::move(x)), 
      atoms(_atoms){}

    #ifdef _WITH_ATEN
    Ptensors(const at::Tensor& T, const AtomsPack& _atoms):
      Ptensors(rtensor::regular(T),_atoms){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors(const Ptensors& x, const int _dev):
      RtensorPackB(x,_dev),
      atoms(x.atoms){}

    Ptensors& to_device(const int _dev){
      RtensorPackB::to_device(_dev);
      return *this;
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    int getn() const{
      return size();
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }
    
    rtensor tensor_of(const int i) const{
      return RtensorPackB::operator()(i);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------

    
    TransferMap<AtomsPack> overlaps(const AtomsPack& x){
      return atoms.overlaps(x);
    }

  };

}

#endif 
