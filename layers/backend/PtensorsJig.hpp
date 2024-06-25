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

#ifndef _ptens_PtensorsJig
#define _ptens_PtensorsJig

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"
#include "ptr_pair_indexed_object_bank.hpp"
#include "observable.hpp"
#include "AtomsPackMatch.hpp"


namespace ptens{


  class PtensorsJig{
  public:

    //shared_ptr<AtomsPackObj> atoms;

    virtual ~PtensorsJig(){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    //PtensorsJig(const shared_ptr<AtomsPackObj>& _atoms):
    //atoms(new AtomsPackObj(*_atoms)){} // this copy is to break the circular dependency 


  public: // ---- Access ------------------------------------------------------------------------------------


    //int size() const{
    //return atoms->size();
    //}

    //int offset1(const int i) const{
    //return atoms->offset(i);
    //}

    //virtual int size_of(const int i) const=0;
    //virtual int offset(const int i) const=0;


  public: // ---- I/O ----------------------------------------------------------------------------------------


    //virtual string str(const string indent="") const=0;


  };

}

#endif 
