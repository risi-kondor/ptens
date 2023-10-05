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

#include "SubgraphLayer.hpp"
#include "SubgraphLayer2.hpp"
#include "EMPlayers2.hpp"


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
    using TLAYER::overlaps;


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


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename TLAYER2>
    SubgraphLayer2(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer2(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),x.dev){
      emp02(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer0<TLAYER2>& x){
      emp02_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer2(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer2(x.G,_S,x.G.subgraphs(_S),5*x.get_nc(),x.dev){
      emp12(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer1<TLAYER2>& x){
      emp12_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer2(const SubgraphLayer2<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer2(x.G,_S,x.G.subgraphs(_S),15*x.get_nc(),x.dev){
      emp22(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer2<TLAYER2>& x){
      emp22_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }


  public: // ---- Message passing from Ptensor layers ---------------------------------------------------------


    SubgraphLayer2(const Ptensors0& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,_G.subgraphs(_S),2*x.get_nc(),x.dev){
      emp02(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors0& x){
      emp02_back(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    SubgraphLayer2(const Ptensors1& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,_G.subgraphs(_S),5*x.get_nc(),x.dev){
      emp12(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors1& x){
      emp12_back(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    SubgraphLayer2(const Ptensors2& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,_G.subgraphs(_S),15*x.get_nc(),x.dev){
      emp22(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors2& x){
      emp22_back(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    SubgraphLayer2 permute(const cnine::permutation& pi){
      return SubgraphLayer2(Ptensors2::permute(pi),G.permute(pi),S);
    }


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
