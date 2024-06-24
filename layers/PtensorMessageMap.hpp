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

#ifndef _ptens_PtensorMessageMap
#define _ptens_PtensorMessageMap

#include "PtensorMessageMapObj.hpp"


namespace ptens{

  class AtomsPackObj;


  class PtensorMessageMap{
  public:
    
    shared_ptr<PtensorMessageMapObj> obj;

    PtensorMessageMap(const shared_ptr<PtensorMessageMapObj>& x):
      obj(x){}

    PtensorMessageMap(PtensorMessageMapObj* x):
      obj(x){}


  public: // ---- Static constructors -----------------------------------------------------------------------------


    static PtensorMessageMap all_overlapping(const AtomsPack& out, const AtomsPack& in){
      return PtensorMessageMap(PtensorMessageMapObj::all_overlapping(*out.obj,*in.obj));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorMessageMap"; 
    }

    string repr() const{
      return "PtensorMessageMap";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const PtensorMessageMap& x){
      stream<<x.str(); return stream;}

  };

}

#endif
