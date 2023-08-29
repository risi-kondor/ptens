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

#ifndef _ptens_SubgraphLayer1
#define _ptens_SubgraphLayer1

#include "Hgraph.hpp"
#include "Subgraph.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "TransferMap.hpp"
#include "EMPlayers2.hpp"

#include "SubgraphLayer0.hpp"


namespace ptens{

  template<typename TLAYER> 
  class SubgraphLayer1: public SubgraphLayer<TLAYER>{
  public:

    typedef cnine::RtensorA rtensor;
    typedef SubgraphLayer<TLAYER> BASE;

    using BASE::BASE;
    //using TLAYER::dev;
    using BASE::atoms;
    //using TLAYER::getn;
    //using TLAYER::get_nc;

    //const Hgraph& G;
    //const Subgraph& S;


  public: 

    //template<typename IPACK>
    //SubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const IPACK& ipack, const int nc, const int _dev=0):
    //G(_G), S(_S), TLAYER(ipack,nc,cnine::fill_zero(),_dev){}


  public: // ---- Message passing ----------------------------------------------------------------------------------------


    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,AtomsPack(FindPlantedSubgraphs(*x.G.obj,*_S.obj)),x.get_nc(),x.dev){
      emp01(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,AtomsPack(FindPlantedSubgraphs(*x.G.obj,*_S.obj)),x.get_nc(),x.dev){
      emp11(*this,x,TransferMap(x.atoms,atoms));
    }


  public: 




  public:




  };

}

#endif 
    //template<typename LAYER>
    //TransferMap overlaps(const LAYER& x){
    //return TransferMap(atoms,x.atoms);
    //}

    //SubgraphLayer0<TLAYER> transfer0(const Subgraph& _S){
    //SubgraphLayer0<TLAYER> R(G,_S,getn(),get_nc());
    //emp10(R,*this,TransferMap(atoms,R.atoms));
    //}

    //SubgraphLayer1<TLAYER> transfer1(const Subgraph& _S){
    //SubgraphLayer1<TLAYER> R(G,_S,FindPlantedSubgraphs(G,_S),get_nc());
    //emp11(R,*this,TransferMap(atoms,R.atoms));
    //}
