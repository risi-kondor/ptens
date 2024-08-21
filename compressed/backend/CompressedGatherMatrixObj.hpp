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

#ifndef _ptens_CompressedGatherMatrixObj
#define _ptens_CompressedGatherMatrixObj

#include "BlockCsparseMatrix.hpp"


namespace ptens{


  class CompressedGatherMatrixObj: public cnine::BlockCsparseMatrix<float>{
  public:

    typedef cnine::BlockCsparseMatrix<float> BASE;

    using BASE::BASE;

    //BSmatrix in_matrix;
    //BSmatrix out_matrix;

    

  };

}

#endif 
