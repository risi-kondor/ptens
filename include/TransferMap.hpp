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

#ifndef _ptens_TransferMap
#define _ptens_TransferMap

#include "TransferMapObj.hpp"


namespace ptens{

  class AtomsPackObj;

  class TransferMap{
  public:
    
    shared_ptr<TransferMapObj<AtomsPackObj> > obj;

    TransferMap(const shared_ptr<TransferMapObj<AtomsPackObj> >& x):
      obj(x){}

    //TransferMap(TransferMapObj<AtomsPackObj>* x):
    //obj(x){}

    //TransferMap(const ATOMSPACK& _in_atoms, const ATOMSPACK& _out_atoms):
    //obj(new TransferMapObj(_in_atoms,_out_atoms){}


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      return obj->is_empty();
    }

    const AindexPack& in() const{
      return *obj->in;
    }

    const AindexPack& out() const{
      return *obj->out;
    }

    std::shared_ptr<cnine::GatherMap> get_bmap() const{
      return obj->get_bmap();
    }
    
    //void for_each_edge(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
    //obj->for_each_edge(lambda,self);
    //}

  };

}

#endif 

