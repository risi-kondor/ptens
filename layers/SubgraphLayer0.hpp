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

#ifndef _ptens_SubgraphLayer0
#define _ptens_SubgraphLayer0

//#include "Hgraph.hpp"
//#include "Subgraph.hpp"
//#include "TransferMap.hpp"
#include "SubgraphLayer.hpp"
#include "EMPlayers2.hpp"


namespace ptens{

  template<typename TLAYER> 
  class SubgraphLayer1;


  template<typename TLAYER> 
  class SubgraphLayer0: public SubgraphLayer<TLAYER>{
  public:

    typedef cnine::RtensorA rtensor;
    typedef SubgraphLayer<TLAYER> BASE;

    using BASE::BASE;
    using TLAYER::dev;
    using TLAYER::atoms;
    using TLAYER::getn;
    using TLAYER::get_nc;
    using TLAYER::tensor;


  public: 

    // 
    //template<typename IPACK>
    //SubgraphLayer0(const Hgraph& _G, const Subgraph& _S, const IPACK& ipack, const int nc, const int _dev=0):
    //G(_G), S(_S), TLAYER(ipack,nc,cnine::fill_zero(),_dev){}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SubgraphLayer0(const Hgraph& _G, const int nc, const FILLTYPE& dummy, const int _dev=0):
    //G(_G), S(Subgraph::trivial()), TLAYER(_G.getn(),nc,dummy,_dev){}

    //SubgraphLayer0(const Hgraph& _G, const rtensor& x):
    //G(_G), S(Subgraph::trivial()), TLAYER(x){}


  public: // ---- Copying ------------------------------------------------------------------------------------------------


    SubgraphLayer0(const SubgraphLayer0& x):
      SubgraphLayer<TLAYER>(x){}


  public: // ---- Message passing ----------------------------------------------------------------------------------------


    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,AtomsPack(x.getn()),x.get_nc(),x.dev){
      emp00(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,AtomsPack(x.getn()),x.get_nc(),x.dev){
      emp10(*this,x,TransferMap(x.atoms,atoms));
    }


  public:


  };

}

#endif 


    // 
    //template<typename LAYER>
    //TransferMap overlaps(const LAYER& x){
    //  return TransferMap(atoms,x.atoms);
    //}


    //SubgraphLayer0<TLAYER> transfer0(const Subgraph& _S){
    //SubgraphLayer0<TLAYER> R(G,_S,AtomsPack(getn()),get_nc(),dev);
    //emp00(R,*this,TransferMap(atoms,R.atoms));
    //}

    //SubgraphLayer1<TLAYER> transfer1(const Subgraph& _S){
    //SubgraphLayer1<TLAYER> R(G,_S,FindPlantedSubgraphs(G,_S),get_nc(),dev);
    //emp11(R,*this,TransferMap(atoms,R.atoms));
    //}

