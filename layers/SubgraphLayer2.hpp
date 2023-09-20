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

#ifndef _ptens_SubgraphLayer2
#define _ptens_SubgraphLayer2

#include "Hgraph.hpp"
#include "Subgraph.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "TransferMap.hpp"
#include "EMPlayers2.hpp"
#include "SubgraphLayer2.hpp"


namespace ptens{

  template<typename TLAYER> 
  class SubgraphLayer0;
  template<typename TLAYER> 
  class SubgraphLayer1;


  template<typename TLAYER> 
  class SubgraphLayer2: public SubgraphLayer<TLAYER>{
  public:

    typedef cnine::RtensorA rtensor;
    typedef SubgraphLayer<TLAYER> BASE;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::G;
    using BASE::S;
    using TLAYER::dev;
    using TLAYER::getn;
    using TLAYER::get_nc;
    using TLAYER::get_grad;
    using TLAYER::inp;
    using TLAYER::diff2;


  public: 

    //template<typename IPACK>
    //SubgraphLayer2(const Ggraph& _G, const Subgraph& _S, const IPACK& ipack, const int nc, const int _dev=0):
    //G(_G), S(_S), TLAYER(ipack,nc,cnine::fill_zero(),_dev){}


  public: // ---- Named Constructors ------------------------------------------------------------------------------------------


    static SubgraphLayer2<TLAYER> zeros_like(const SubgraphLayer2<TLAYER>& x){
      return SubgraphLayer2(TLAYER::zeros_like(x),x.G,x.S);
    }

    static SubgraphLayer2<TLAYER> randn_like(const SubgraphLayer2<TLAYER>& x){
      return SubgraphLayer2(TLAYER::randn_like(x),x.G,x.S);
    }

    SubgraphLayer2<TLAYER> zeros() const{
      return SubgraphLayer2(TLAYER::zeros_like(*this),G,S);
    }

    SubgraphLayer2<TLAYER> zeros(const int _nc) const{
      return SubgraphLayer2(TLAYER::zeros_like(*this,_nc),G,S);
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    SubgraphLayer2(const SubgraphLayer2<TLAYER>& x, const int _dev):
      SubgraphLayer<TLAYER>(TLAYER(x,_dev),x.G,x.S){}


  public: // ---- Message passing ----------------------------------------------------------------------------------------


    template<typename TLAYER2>
    SubgraphLayer2(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      //SubgraphLayer2(x.G,_S,AtomsPack(CachedPlantedSubgraphs()(*x.G.obj,*_S.obj)),2*x.get_nc(),x.dev){
      SubgraphLayer2(x.G,_S,CachedPlantedSubgraphsMx(*x.G.obj,*_S.obj),2*x.get_nc(),x.dev){
      emp02(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer0<TLAYER2>& x){
      emp02_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer2(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      //SubgraphLayer2(x.G,_S,AtomsPack(CachedPlantedSubgraphs()(*x.G.obj,*_S.obj)),5*x.get_nc(),x.dev){
      SubgraphLayer2(x.G,_S,CachedPlantedSubgraphsMx(*x.G.obj,*_S.obj),5*x.get_nc(),x.dev){
      emp12(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer1<TLAYER2>& x){
      emp12_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer2(const SubgraphLayer2<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer2(x.G,_S,CachedPlantedSubgraphsMx(*x.G.obj,*_S.obj),15*x.get_nc(),x.dev){
      emp22(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer2<TLAYER2>& x){
      emp22_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms));
    }


    SubgraphLayer2(const Ptensors0& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,CachedPlantedSubgraphsMx(*_G.obj,*_S.obj),2*x.get_nc(),x.dev){
      emp02(*this,x,TransferMap(x.atoms,atoms));
    }

    void gather_back(Ptensors0& x){
      emp20(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms)); 
    }

    SubgraphLayer2(const Ptensors1& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,CachedPlantedSubgraphsMx(*_G.obj,*_S.obj),5*x.get_nc(),x.dev){
      emp12(*this,x,TransferMap(x.atoms,atoms));
    }

    void gather_back(Ptensors1& x){
      emp21(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms)); 
    }

    SubgraphLayer2(const Ptensors2& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,CachedPlantedSubgraphsMx(*_G.obj,*_S.obj),15*x.get_nc(),x.dev){
      emp22(*this,x,TransferMap(x.atoms,atoms));
    }

    void gather_back(Ptensors2& x){
      emp22(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms)); 
    }


  public: 




  public:


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SubgraphLayer2";
    }

    string repr() const{
      if(dev==0) return "<SubgraphLayer2[N="+to_string(getn())+"]>";
      else return "<SubgraphLayer2[N="+to_string(getn())+"][G]>";
    }



  };

}

#endif 
