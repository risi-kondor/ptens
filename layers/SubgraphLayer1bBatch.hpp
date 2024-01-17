/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _ptens_SubgraphLayer0bBatch
#define _ptens_SubgraphLayer0bBatch

#include "GgraphBatch.hpp"
#include "SubgraphLayer1b.hpp"
#include "Ptensors1bBatch.hpp"


namespace ptens{


  template<typename TYPE> 
  class SubgraphLayer1b: public Ptensors1bBatch<TYPE>{
  public:

    typedef Ptensors1bBatch<TYPE> BASE;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const GgraphBatch G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    SubgraphLayer1bBatch(const GgraphBatch& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer1bBatch(const GgraphBatch& _G, const Subgraph& _S, const AtomsPackBatch& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    SubgraphLayer1b copy() const{
      return SubgraphLayer1b(G,S,BASE::copy());
    }

    SubgraphLayer1b copy(const int _dev) const{
      return SubgraphLayer1b(G,S,BASE::copy(_dev));
    }

    SubgraphLayer1b zeros_like() const{
      return SubgraphLayer1b(G,S,BASE::zeros_like());
    }

    SubgraphLayer1b gaussian_like() const{
      return SubgraphLayer1b(G,S,BASE::gaussian_like());
    }

    static SubgraphLayer1b* new_zeros_like(const SubgraphLayer1b& x){
      return new SubgraphLayer1b(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    SubgraphLayer1b(const SubgraphLayer1b& x, const int _dev):
      SubgraphLayer1b(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    //static SubgraphLayer1bBatch<TYPE> linmaps(const SOURCE& x){
    //Ptensors1bBatch<TYPE> R;
    //for(int i=0; i<x.size(); i++)
    //R.obj.push_back(PtensorsOb<TYPE>::linmaps(x[i]));
    //return R;
    //}

    template<typename SOURCE>
    SubgraphLayer1b(const SOURCE& x, const Subgraph& _S):
      SubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x);
    }

    template<typename SOURCE>
    SubgraphLayer1b(const SOURCE& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1b(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x);
    }

  };

}

#endif 
