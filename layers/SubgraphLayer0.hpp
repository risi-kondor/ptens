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

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
//#include "SubgraphLayer.hpp"


namespace ptens{


  template<typename TYPE> 
  class SubgraphLayer0: public Ptensors0<TYPE>{
  public:

    typedef Ptensors0<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    //typedef cnine::Ltensor<TYPE> TENSOR;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const Ggraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    SubgraphLayer0(const Ggraph& _G, const TENSOR& x):
      BASE(x), G(_G), S(Subgraph::trivial()){}

    SubgraphLayer0(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer0(const Ggraph& _G, const Subgraph& _S, const AtomsPack& atoms, const TENSOR& x):
      BASE(x,atoms), G(_G), S(_S){}

    SubgraphLayer0(const Ggraph& _G, const int nc, const int fcode=0, const int _dev=0):
      G(_G), S(Subgraph::trivial()), BASE(_G.getn(),nc,fcode,_dev){}

    SubgraphLayer0(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    //static SubgraphLayer0 cat(const vector<SubgraphLayer0>& list){
    //vector<AtomsPack0> v;
    //for(auto& p:list)
    //v.push_back(p.atoms);
    //return SubgraphLayer0(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    //}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    SubgraphLayer0 copy() const{
      return SubgraphLayer0(G,S,BASE::copy());
    }

    SubgraphLayer0 copy(const int _dev) const{
      return SubgraphLayer0(G,S,BASE::copy(_dev));
    }

    SubgraphLayer0 zeros_like() const{
      return SubgraphLayer0(G,S,BASE::zeros_like());
    }

    SubgraphLayer0 gaussian_like() const{
      return SubgraphLayer0(G,S,BASE::gaussian_like());
    }

    static SubgraphLayer0* new_zeros_like(const SubgraphLayer0& x){
      return new SubgraphLayer0(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    SubgraphLayer0(const SubgraphLayer0& x, const int _dev):
      SubgraphLayer0(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    // TODO 
    template<typename SOURCE>
    static SubgraphLayer0 linmaps(const SOURCE& x){
      SubgraphLayer0 R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    SubgraphLayer0(const SOURCE& x, const Subgraph& _S):
      SubgraphLayer0(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
    }

    template<typename SOURCE>
    SubgraphLayer0(const SOURCE& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer0(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
    }

  };


  template<typename SOURCE>
  inline SubgraphLayer0<float> sglinmaps0(const SOURCE& x){
    SubgraphLayer0<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer0<float> gather0(const SOURCE& x, const Subgraph& _S){
    int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
    SubgraphLayer0<float> R(x.G,_S,x.G.subgraphs(_S),nc,0,x.dev);
    R.add_gather(x,LayerMap::overlaps_map(R.atoms,x.atoms));
    return R;
  }


}

#endif 


