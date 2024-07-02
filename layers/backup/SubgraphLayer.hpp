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

#ifndef _ptens_SubgraphLayer
#define _ptens_SubgraphLayer

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "TransferMap.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "NodeLayer.hpp"


namespace ptens{

  template<typename TLAYER> 
  class SubgraphLayer: public TLAYER{
  public:

    typedef cnine::RtensorA rtensor;

    using TLAYER::TLAYER;
    using TLAYER::dev;
    using TLAYER::atoms;
    using TLAYER::get_nc;
    using TLAYER::tensor;

    const Ggraph G;
    const Subgraph S;


  public: 

    SubgraphLayer(){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SubgraphLayer(const Ggraph& _G, const int nc, const FILLTYPE& dummy, const int _dev=0):
    G(_G), S(Subgraph::trivial()), TLAYER(_G.getn(),nc,dummy,_dev){}

    template<typename IPACK>
    SubgraphLayer(const Ggraph& _G, const Subgraph& _S, const IPACK& ipack, const int nc, const int _dev=0):
      G(_G), S(_S), TLAYER(ipack,nc,cnine::fill_zero(),_dev){}

    template<typename IPACK>
    SubgraphLayer(const Ggraph& _G, const Subgraph& _S, const IPACK& _ipack, const int _nc, const cnine::fill_raw& dummy, const int nc, const int _dev=0):
      G(_G), S(_S), TLAYER(TLAYER::raw(_ipack,_nc,_dev)){}

    template<typename IPACK>
    SubgraphLayer(const Ggraph& _G, const Subgraph& _S, const IPACK& _ipack, const int _nc, const cnine::fill_zero& dummy, const int nc, const int _dev=0):
      G(_G), S(_S), TLAYER(TLAYER::zero(_ipack,_nc,_dev)){}

    template<typename IPACK>
    SubgraphLayer(const Ggraph& _G, const Subgraph& _S, const IPACK& _ipack, const int _nc, const cnine::fill_gaussian& dummy, const int nc, const int _dev=0):
      G(_G), S(_S), TLAYER(TLAYER::gaussian(_ipack,_nc,_dev)){}

    template<typename IPACK>
    SubgraphLayer(const Ggraph& _G, const Subgraph& _S, const IPACK& _ipack, const int _nc, const cnine::fill_sequential& dummy, const int nc, const int _dev=0):
      G(_G), S(_S), TLAYER(TLAYER::sequential(_ipack,_nc,_dev)){}


  public: // ---- Copying ------------------------------------------------------------------------------------------------


    SubgraphLayer(const SubgraphLayer<TLAYER>& x):
      TLAYER(x), G(x.G), S(x.S){}

    
  public: // ---- Transport ----------------------------------------------------------------------------------


    //SubgraphLayer(const SubgraphLayer<TLAYER>& x, const int _dev):
    //TLAYER(x,_dev), G(x.G), S(x.S){}


  public: // ---- Conversions --------------------------------------------------------------------------------------------


    SubgraphLayer(TLAYER&& x, const Ggraph& _G, const Subgraph& _S):
      TLAYER(std::move(x)), G(_G), S(_S){}


  public: // ---- Operations ---------------------------------------------------------------------------------------------


    void add(const SubgraphLayer<TLAYER>& x){
      PTENS_ASSRT(G==x.G);
      PTENS_ASSRT(S==x.S);
      TLAYER::add(x);
    }


  public: // ---- Message passing ----------------------------------------------------------------------------------------



  public:


  };

}

#endif 
