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

#ifndef _ptens_GatherPlan
#define _ptens_GatherPlan

#include "GatherPlanObj.hpp"
#include "AtomsPack.hpp"


namespace ptens{


  class GatherPlan{
  public:
    
    shared_ptr<GatherPlanObj> obj;

    GatherPlan(){
      PTENS_ASSRT(false);}

    GatherPlan(const shared_ptr<GatherPlanObj>& x):
      obj(x){}

    const AindexPackB& in() const{
      return *obj->in_map;
    }

    const AindexPackB& out() const{
      return *obj->out_map;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "GatherPlan";
    }

    string repr() const{
      return "GatherPlan";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const GatherPlan& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
