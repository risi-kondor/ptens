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

#ifndef _ptens_GatherPlanObj
#define _ptens_GatherPlanObj

#include "observable.hpp"
#include "AindexPackB.hpp"
#include "GatherMapB.hpp"
#include "flog.hpp"


namespace ptens{


  class GatherPlanObj: public cnine::observable<GatherPlanObj>{
  public:

    shared_ptr<AindexPackB> out_map;
    shared_ptr<AindexPackB> in_map;

    GatherPlanObj():
      observable(this){}

    GatherPlanObj(const shared_ptr<AindexPackB>& out, const shared_ptr<AindexPackB>& in):
      observable(this),
      out_map(out),
      in_map(in){}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "GatherPlanObj";
    }

    string repr() const{
      return "<GatherPlanObj>";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"In:"<<endl;
      oss<<in_map->str(indent+"  ");
      oss<<"Out:"<<endl;
      oss<<out_map->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherPlanObj& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
