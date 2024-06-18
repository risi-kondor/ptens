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

#include "SubgraphLayer.hpp"
#include "NodeLayer.hpp"
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


    SubgraphLayer0(const NodeLayer& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,x.G.subgraphs(_S),x.get_nc(),x.dev){
	x.emp_to(*this);
    }

    void gather_back(NodeLayer& x){
      x.get_grad().emp_from(get_grad());
    }

    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,x.G.subgraphs(_S),x.get_nc(),x.dev){
      emp00(*this,x,overlaps(x.atoms));
    }
    
    template<typename TLAYER2>
    void gather_back(SubgraphLayer0<TLAYER2>& x){
      emp00(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,x.G.subgraphs(_S),x.get_nc(),x.dev){
      emp10(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer1<TLAYER2>& x){
      emp01(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer0(const SubgraphLayer2<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),x.dev){
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
