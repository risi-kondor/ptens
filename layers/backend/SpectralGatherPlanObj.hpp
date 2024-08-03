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

#ifndef _ptens_SpectralGatherPlanObj
#define _ptens_SpectralGatherPlanObj

#include "GatherPlanObj.hpp"
#include "BlockCsparseMatrix.hpp"


namespace ptens{


  class SpectralGatherPlanObj: public GatherPlanObj{
  public:

    cnine::BlockCsparseMatrix BMATRIX;

    BMATRIX matrix;

  };

}

#endif 
