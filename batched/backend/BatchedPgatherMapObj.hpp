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

#ifndef _ptens_BatchedPgatherMapObj
#define _ptens_BatchedPgatherMapObj

#include "observable.hpp"
#include "BatchedAindexPackB.hpp"
#include "GatherMapB.hpp"
#include "flog.hpp"


namespace ptens{


  class BatchedPgatherMapObj: public cnine::observable<BatchedPgatherMapObj>{
  public:

    shared_ptr<BatchedAindexPackB> out_map;
    shared_ptr<BatchedAindexPackB> in_map;


    BatchedPgatherMapObj():
      observable(this){}

    BatchedPgatherMapObj(const shared_ptr<BatchedAindexPackB>& out, const shared_ptr<BatchedAindexPackB>& in):
      observable(this),
      out_map(out),
      in_map(in){}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedPgatherMapObj";
    }

    string repr() const{
      return "<BatchedPgatherMapObj>";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"In:"<<endl;
      oss<<in_map->str(indent+"  ");
      oss<<"Out:"<<endl;
      oss<<out_map->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPgatherMapObj& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
