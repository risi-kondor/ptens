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
#ifndef _ptens_Ptensor
#define _ptens_Ptensor

#include "Atoms.hpp"
#include "RtensorObj.hpp"

#define PTENSOR_PTENSOR_IMPL cnine::RtensorObj


namespace ptens{

  class Ptensor: public PTENSOR_PTENSOR_IMPL{
  public:

    Atoms atoms;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor(const int _k, const Atoms& _atoms, const FILLTYPE& dummy, const int _dev=0):
      PTENSOR_PTENSOR_IMPL(vector<int>(_k,_atoms.size()),dummy,_dev),
      atoms(_atoms){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor raw(const int _k, const Atoms& _atoms, const int _dev=0){
	return Ptensor(_k,_atoms,cnine::fill_raw(),_dev);}

    static Ptensor zero(const int _k, const Atoms& _atoms, const int _dev=0){
	return Ptensor(_k,_atoms,cnine::fill_zero(),_dev);}

    static Ptensor gaussian(const int _k, const Atoms& _atoms, const int _dev=0){
	return Ptensor(_k,_atoms,cnine::fill_gaussian(),_dev);}

    


  };


}


#endif 
