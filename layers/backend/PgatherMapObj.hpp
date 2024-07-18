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

#ifndef _ptens_PgatherMapObj
#define _ptens_PgatherMapObj

#include "observable.hpp"
#include "AindexPackB.hpp"
#include "GatherMapB.hpp"
#include "flog.hpp"


namespace ptens{


  class PgatherMapObj: public cnine::observable<PgatherMapObj>{
  public:

    shared_ptr<AindexPackB> in_map;
    shared_ptr<AindexPackB> out_map;

    PgatherMapObj():
      observable(this){}
    //in_map(new AindexPackB()),
    //out_map(new AindexPackB()){}



  };

}


#endif 
