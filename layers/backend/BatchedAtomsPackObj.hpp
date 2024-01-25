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
 */


#ifndef _ptens_BatchedAtomsPackObj
#define _ptens_BatchedAtomsPackObj

#include "object_pack_s.hpp"
#include "AtomsPackObj.hpp"


namespace ptens{


  class BatchedAtomsPackObj: public cnine::object_pack_s<AtomsPackObj>{
  public:

    typedef cnine::object_pack_s<AtomsPackObj> BASE;

    using BASE::BASE;
    using BASE::size;
    using BASE::operator[];


  };

}

#endif 
