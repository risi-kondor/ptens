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

#ifndef _ptens_CompressedSubgraphLayer1
#define _ptens_CompressedSubgraphLayer1

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "CompressedPtensors1.hpp"
//#include "CompressedSubgraphLayer.hpp"
//#include "Rtensor3_view.hpp"


namespace ptens{


  template<typename TYPE> 
  class CompressedSubgraphLayer1: public CompressedPtensors1<TYPE>{
  public:

    typedef CompressedPtensors1<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;
    using BASE::size;
    using BASE::get_nc;
    using BASE::get_dev;
    using BASE::get_grad;
    using BASE::view3;
    using BASE::cols;

    using TENSOR::dim;
    using TENSOR::get_arr;

    const Ggraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    CompressedSubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    //CompressedSubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const AtomsPack& atoms, const TENSOR& x):
    //BASE(atoms,x), G(_G), S(_S){}

    CompressedSubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_G.subgraphs(_S),nc,fcode,_dev){}

    CompressedSubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    CompressedSubgraphLayer1(const SubgraphLayer1& x, const int _dev):
      SubgraphLayer1(x.G,x.S,BASE(x,_dev)){}

  };

}

#endif 
