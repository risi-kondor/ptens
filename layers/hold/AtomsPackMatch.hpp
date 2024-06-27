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

#ifndef _ptens_AtomsPackMatch
#define _ptens_AtomsPackMatch

#include "AtomsPackObj.hpp"
#include "AtomsPackMatchObj.hpp"
#include "observable.hpp"

namespace ptens{


  class AtomsPackMatch{
  public:

    shared_ptr<const AtomsPackMatchObj> obj;

    ~AtomsPackMatch(){}

    AtomsPackMatch(){}
    
    AtomsPackMatch(const AtomsPackMatchObj* _obj):
      obj(_obj){}


  public: // ---- Named constructors ------------------------------------------------------------------------


    static AtomsPackMatch overlaps(const cnine::array_pool<int>& in_atoms, 
      const cnine::array_pool<int>& out_atoms){
      return AtomsPackMatch(new AtomsPackMatchObj(in_atoms,out_atoms));
    }

    static AtomsPackMatch overlaps(const cnine::array_pool<int>& in_atoms, 
      const cnine::array_pool<int>& out_atoms, const int min_overlaps){
      return AtomsPackMatch(new AtomsPackMatchObj(in_atoms,out_atoms,min_overlaps));
    }

    pair<const cnine::hlists<int>&, const cnine::hlists<int>&> lists() const{
      return pair<const cnine::hlists<int>&, const cnine::hlists<int>&>(obj->in,obj->out);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPackMatch";
    }

    string repr() const{
      return "AtomsPackMatch";
    }

    string str(const string indent="") const{
      return obj->str();
    }

    friend ostream& operator<<(ostream& stream, const AtomsPackMatch& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
