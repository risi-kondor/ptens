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

#ifndef _ptens_BatchedGatherPlan
#define _ptens_BatchedGatherPlan

#include "BatchedGatherPlanObj.hpp"


namespace ptens{


  class BatchedGatherPlan{
  public:
    
    shared_ptr<BatchedGatherPlanObj> obj;

    BatchedGatherPlan(){
      PTENS_ASSRT(false);}

    BatchedGatherPlan(const shared_ptr<BatchedGatherPlanObj>& x):
      obj(x){}

    const BatchedAindexPackB& in() const{
      return *obj->in_map;
    }

    const BatchedAindexPackB& out() const{
      return *obj->out_map;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedGatherPlan";
    }

    string repr() const{
      return "BatchedGatherPlan";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedGatherPlan& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
