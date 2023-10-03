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
#include "PtensFindPlantedSubgraphs.hpp"
#include "EMPlayers2.hpp"


namespace ptens{

  template<typename TLAYER> 
  class SubgraphLayer1;
  template<typename TLAYER> 
  class SubgraphLayer2;


  template<typename TLAYER> 
  class SubgraphLayer0: public SubgraphLayer<TLAYER>{
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
    using TLAYER::tensor;
    using TLAYER::inp;
    using TLAYER::diff2;
    using TLAYER::inv_channel_norms;
    using TLAYER::add_scale_channels;
    using TLAYER::overlaps;



  public: // ---- Named Constructors ------------------------------------------------------------------------------------------


    static SubgraphLayer0<TLAYER> zeros_like(const SubgraphLayer0<TLAYER>& x){
      return SubgraphLayer0(TLAYER::zeros_like(x),x.G,x.S);
    }

    static SubgraphLayer0<TLAYER> randn_like(const SubgraphLayer0<TLAYER>& x){
      return SubgraphLayer0(TLAYER::randn_like(x),x.G,x.S);
    }

    SubgraphLayer0<TLAYER> zeros() const{
      return SubgraphLayer0(TLAYER::zeros_like(*this),G,S);
    }

    SubgraphLayer0<TLAYER> zeros(const int _nc) const{
      return SubgraphLayer0(TLAYER::zeros_like(*this,_nc),G,S);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------------------


    SubgraphLayer0(const Ggraph& _G, const rtensor& x):
      SubgraphLayer<TLAYER>(TLAYER(x),_G,Subgraph::trivial()){}


  public: // ---- Transport ----------------------------------------------------------------------------------


    SubgraphLayer0(const SubgraphLayer0<TLAYER>& x, const int _dev):
      SubgraphLayer<TLAYER>(TLAYER(x,_dev),x.G,x.S){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,AtomsPack(x.getn()),x.get_nc(),x.dev){
      emp00(*this,x,overlaps(x.atoms));
    }
    
    template<typename TLAYER2>
    void gather_back(SubgraphLayer0<TLAYER2>& x){
      emp00(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,AtomsPack(x.getn()),x.get_nc(),x.dev){
      emp10(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer1<TLAYER2>& x){
      emp01(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer2<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,AtomsPack(x.getn()),2*x.get_nc(),x.dev){
      emp20(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer2<TLAYER2>& x){
      emp20_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }


  public: // ---- Message passing from Ptensor layers ---------------------------------------------------------


    SubgraphLayer0(const Ptensors0& x, const Ggraph& g, const Subgraph& s):
      SubgraphLayer0(g,s,g.subgraphs(s),x.get_nc(),x.dev){
      emp00(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors0& x){
      emp00(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    SubgraphLayer0(const Ptensors1& x, const Ggraph& g, const Subgraph& s):
      SubgraphLayer0(g,s,g.subgraphs(s),x.get_nc(),x.dev){
      emp10(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors1& x){
      emp01(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    SubgraphLayer0(const Ptensors2& x, const Ggraph& g, const Subgraph& s):
      SubgraphLayer0(g,s,g.subgraphs(s),2*x.get_nc(),x.dev){
      emp20(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors2& x){
      emp20_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    SubgraphLayer0 permute(const cnine::permutation& pi){
      return SubgraphLayer0(Ptensors0::permute(pi),G.permute(pi),S);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SubgraphLayer0";
    }

    string repr() const{
      if(dev==0) return "<SubgraphLayer0[N="+to_string(getn())+"]>";
      else return "<SubgraphLayer0[N="+to_string(getn())+"][G]>";
    }


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

