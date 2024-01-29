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

#ifndef _ptens_BatchedSubgraphLayer1b
#define _ptens_BatchedSubgraphLayer1b

#include "BatchedGgraph.hpp"
#include "Subgraph.hpp"
#include "BatchedPtensors1b.hpp"


namespace ptens{


  template<typename TYPE> 
  class BatchedSubgraphLayer1b: public BatchedPtensors1b<TYPE>{
  public:

    typedef BatchedPtensors1b<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    typedef BatchedAtomsPackN<AtomsPack1obj<int> > BatchedAtomsPack1;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const BatchedGgraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedSubgraphLayer0b(const BatchedGgraph& _G, const TENSOR& x):
    //BASE(x), G(_G), S(Subgraph::trivial()){}

    //BatchedSubgraphLayer0b(const Ggraph& _G, const Subgraph& _S, const BASE& x):
    //BASE(x), G(_G), S(_S){}

    BatchedSubgraphLayer1b(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack1& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    BatchedSubgraphLayer1b(const BatchedGgraph& _G, const int nc, const int fcode=0, const int _dev=0):
      G(_G), S(Subgraph::trivial()), BASE(_G.getn(),nc,fcode,_dev){}

    BatchedSubgraphLayer1b(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    /*
    static BatchedSubgraphLayer0b cat(const vector<BatchedSubgraphLayer0b>& list){
      vector<AtomsPack0> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return BatchedSubgraphLayer0b(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }
    */


  public: // ----- Spawning ----------------------------------------------------------------------------------


    BatchedSubgraphLayer1b copy() const{
      return BatchedSubgraphLayer1b(G,S,BASE::copy());
    }

    BatchedSubgraphLayer1b copy(const int _dev) const{
      return BatchedSubgraphLayer1b(G,S,BASE::copy(_dev));
    }

    BatchedSubgraphLayer1b zeros_like() const{
      return BatchedSubgraphLayer1b(G,S,BASE::zeros_like());
    }

    BatchedSubgraphLayer1b gaussian_like() const{
      return BatchedSubgraphLayer1b(G,S,BASE::gaussian_like());
    }

    static BatchedSubgraphLayer1b* new_zeros_like(const BatchedSubgraphLayer1b& x){
      return new BatchedSubgraphLayer1b(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    //BatchedSubgraphLayer0b(const BatchedSubgraphLayer0b& x, const int _dev):
    //BatchedSubgraphLayer0b(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    static BatchedSubgraphLayer1b linmaps(const SOURCE& x){
      BatchedSubgraphLayer1b R(x.G,x.S,x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    BatchedSubgraphLayer1b(const SOURCE& x, const Subgraph& _S):
      BatchedSubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev){
      add_gather(x);
    }

    template<typename SOURCE>
    BatchedSubgraphLayer1b(const SOURCE& x, const BatchedGgraph& _G, const Subgraph& _S):
      BatchedSubgraphLayer1b(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev){
      add_gather(x);
    }

  };




}

#endif 


