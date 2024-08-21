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

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "SubgraphLayer1.hpp"
#include "Ptensors2.hpp"


namespace ptens{


  template<typename TYPE> 
  class SubgraphLayer2: public Ptensors2<TYPE>{
  public:

    typedef Ptensors2<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    //typedef cnine::Ltensor<TYPE> TENSOR;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const Ggraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    SubgraphLayer2(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer2(const Ggraph& _G, const Subgraph& _S, const AtomsPack& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    SubgraphLayer2(const Ggraph& _G, const Subgraph& _S, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_G.subgraphs(_S),nc,fcode,_dev){}

    SubgraphLayer2(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    //static SubgraphLayer2 cat(const vector<SubgraphLayer2>& list){
    //vector<AtomsPack2> v;
    //for(auto& p:list)
    //v.push_back(p.atoms);
    //return SubgraphLayer2(AtomsPack2::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    //}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    SubgraphLayer2 copy() const{
      return SubgraphLayer2(G,S,BASE::copy());
    }

    SubgraphLayer2 copy(const int _dev) const{
      return SubgraphLayer2(G,S,BASE::copy(_dev));
    }

    SubgraphLayer2 zeros_like() const{
      return SubgraphLayer2(G,S,BASE::zeros_like());
    }

    SubgraphLayer2 gaussian_like() const{
      return SubgraphLayer2(G,S,BASE::gaussian_like());
    }

    static SubgraphLayer2* new_zeros_like(const SubgraphLayer2& x){
      return new SubgraphLayer2(x.zeros_like());
      //return new SubgraphLayer2(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    SubgraphLayer2(const SubgraphLayer2& x, const int _dev):
      SubgraphLayer2(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    static SubgraphLayer2<float> linmaps(const SOURCE& x){
      SubgraphLayer2<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    SubgraphLayer2(const SOURCE& x, const Subgraph& _S):
      SubgraphLayer2(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.dev){
      add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
    }

    template<typename SOURCE>
    SubgraphLayer2(const SOURCE& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer2(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.dev){
      add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
    }



  };

  template<typename SOURCE>
  inline SubgraphLayer2<float> sglinmaps2(const SOURCE& x){
    SubgraphLayer2<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer2<float> gather2(const SOURCE& x, const Subgraph& _S){
    SubgraphLayer2<float> R(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.dev);
    R.add_gather(x,LayerMap::overlaps_map(R.atoms,x.atoms));
    return R;
  }



}

#endif 

