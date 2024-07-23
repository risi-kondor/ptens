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

#ifndef _ptens_BatchedGatherPlanObj
#define _ptens_BatchedGatherPlanObj

#include "observable.hpp"
#include "BatchedAindexPackB.hpp"
#include "GatherMapB.hpp"
#include "flog.hpp"


namespace ptens{


  class BatchedGatherPlanObj: public cnine::observable<BatchedGatherPlanObj>{
  public:

    shared_ptr<BatchedAindexPackB> out_map;
    shared_ptr<BatchedAindexPackB> in_map;


    BatchedGatherPlanObj():
      observable(this){}

    BatchedGatherPlanObj(const shared_ptr<BatchedAindexPackB>& out, const shared_ptr<BatchedAindexPackB>& in):
      observable(this),
      out_map(out),
      in_map(in){}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedGatherPlanObj";
    }

    string repr() const{
      return "<BatchedGatherPlanObj>";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"In:"<<endl;
      oss<<in_map->str(indent+"  ");
      oss<<"Out:"<<endl;
      oss<<out_map->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedGatherPlanObj& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
