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

#ifndef _ptens_AtomsPackObj0
#define _ptens_AtomsPackObj0

#include "AtomsPackObj.hpp"

namespace ptens{

  class AtomsPackObj0{
  public:

    shared_ptr<AtomsPackObj> atoms;


    CompoundTransferMapObj transfer_map(const AtomsPackObj0& y){
      TransferMap tmap=atoms.overlaps(y.atoms);

      Ltensor<int> A(dims(tmap.ntotal(),2));
      int i=0;
      tmap.for_each_edge([&](const int target, const int source, const float v){
	  A.set(i,0,target);
	  A.set(i,1,source);
	  i++;
	});

      CompoundTransferMapObj R;
      R.nodes.push_back(new CTMnode(new GatherMap(A)));
      return R;
    }
    
  };

}

#endif 
