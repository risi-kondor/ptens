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

#ifndef _ptens_BatchedSubgraphLayer2
#define _ptens_BatchedSubgraphLayer2

#include "BatchedGgraph.hpp"
#include "Subgraph.hpp"
#include "BatchedPtensors2.hpp"


namespace ptens{


  template<typename TYPE> 
  class BatchedSubgraphLayer2: public BatchedPtensors2<TYPE>{
  public:

    typedef BatchedPtensors2<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    //typedef BatchedAtomsPackN<AtomsPack2obj<int> > BatchedAtomsPack2;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const BatchedGgraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedSubgraphLayer0(const BatchedGgraph& _G, const TENSOR& x):
    //BASE(x), G(_G), S(Subgraph::trivial()){}

    BatchedSubgraphLayer2(const BatchedGgraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    BatchedSubgraphLayer2(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    BatchedSubgraphLayer2(const BatchedGgraph& _G, const int nc, const int fcode=0, const int _dev=0):
      G(_G), S(Subgraph::trivial()), BASE(_G.getn(),nc,fcode,_dev){}

    BatchedSubgraphLayer2(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    /*
    static BatchedSubgraphLayer0 cat(const vector<BatchedSubgraphLayer0>& list){
      vector<AtomsPack0> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return BatchedSubgraphLayer0(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }
    */


  public: // ----- Spawning ----------------------------------------------------------------------------------


    BatchedSubgraphLayer2 copy() const{
      return BatchedSubgraphLayer2(G,S,BASE::copy());
    }

    BatchedSubgraphLayer2 copy(const int _dev) const{
      return BatchedSubgraphLayer2(G,S,BASE::copy(_dev));
    }

    BatchedSubgraphLayer2 zeros_like() const{
      return BatchedSubgraphLayer2(G,S,BASE::zeros_like());
    }

    BatchedSubgraphLayer2 gaussian_like() const{
      return BatchedSubgraphLayer2(G,S,BASE::gaussian_like());
    }

    static BatchedSubgraphLayer2* new_zeros_like(const BatchedSubgraphLayer2& x){
      return new BatchedSubgraphLayer2(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    //BatchedSubgraphLayer0(const BatchedSubgraphLayer0& x, const int _dev):
    //BatchedSubgraphLayer0(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    static BatchedSubgraphLayer2 linmaps(const SOURCE& x){
      BatchedSubgraphLayer2 R(x.G,x.S,x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    BatchedSubgraphLayer2(const SOURCE& x, const Subgraph& _S):
      BatchedSubgraphLayer2(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.dev){
      add_gather(x);
    }

    template<typename SOURCE>
    BatchedSubgraphLayer2(const SOURCE& x, const BatchedGgraph& _G, const Subgraph& _S):
      BatchedSubgraphLayer2(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.dev){
      add_gather(x);
    }

  };




}

#endif 


