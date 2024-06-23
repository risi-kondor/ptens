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

#ifndef _ptens_BatchedSubgraphLayer0b
#define _ptens_BatchedSubgraphLayer0b

#include "BatchedGgraph.hpp"
#include "Subgraph.hpp"
#include "BatchedPtensors0b.hpp"
//#include "Ptensors1b.hpp"
//#include "Ptensors2b.hpp"
//#include "BatchedSubgraphLayerb.hpp"


namespace ptens{


  template<typename TYPE> 
  class BatchedSubgraphLayer0b: public BatchedPtensors0b<TYPE>{
  public:

    typedef BatchedPtensors0b<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    typedef BatchedAtomsPackN<AtomsPack0obj<int> > BatchedAtomsPack0;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const BatchedGgraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedSubgraphLayer0b(const BatchedGgraph& _G, const TENSOR& x):
    //BASE(x), G(_G), S(Subgraph::trivial()){}

    BatchedSubgraphLayer0b(const BatchedGgraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    BatchedSubgraphLayer0b(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack0& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    //BatchedSubgraphLayer0b(const BatchedGgraph& _G, const int nc, const int fcode=0, const int _dev=0):
    //G(_G), S(Subgraph::trivial()), BASE(_G.getn(),nc,fcode,_dev){}

    BatchedSubgraphLayer0b(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    /*
    static BatchedSubgraphLayer0b cat(const vector<BatchedSubgraphLayer0b>& list){
      vector<AtomsPack0> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return BatchedSubgraphLayer0b(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }
    */

    static BatchedSubgraphLayer0b from_vertex_features(const vector<int>& graphs, const TENSOR& M){
      BatchedGgraph G(graphs);
      vector<int> sizes;
      for(int i=0; i<G.size(); i++)
	sizes.push_back(G[i].getn());
      return BatchedSubgraphLayer0b(G,Subgraph::trivial(),BASE(M,sizes));
    }

    static BatchedSubgraphLayer0b from_edge_features(const vector<int>& graphs, const TENSOR& M){
      BatchedGgraph G(graphs);
      auto atoms=new BatchedAtomsPackNobj<AtomsPack0obj<int> >();
      for(int i=0; i<G.size(); i++)
	atoms->obj.push_back(to_share(new AtomsPack0obj<int>(G[i].original_edges())));
      atoms->make_row_offsets();
      return BatchedSubgraphLayer0b(G,Subgraph::edge(),BASE(BatchedAtomsPack0(atoms),M));
    }


  public: // ----- Spawning ----------------------------------------------------------------------------------


    BatchedSubgraphLayer0b copy() const{
      return BatchedSubgraphLayer0b(G,S,BASE::copy());
    }

    BatchedSubgraphLayer0b copy(const int _dev) const{
      return BatchedSubgraphLayer0b(G,S,BASE::copy(_dev));
    }

    BatchedSubgraphLayer0b zeros_like() const{
      return BatchedSubgraphLayer0b(G,S,BASE::zeros_like());
    }

    BatchedSubgraphLayer0b gaussian_like() const{
      return BatchedSubgraphLayer0b(G,S,BASE::gaussian_like());
    }

    static BatchedSubgraphLayer0b* new_zeros_like(const BatchedSubgraphLayer0b& x){
      return new BatchedSubgraphLayer0b(x.G,x.S,x.BASE::zeros_like());
    }
    
    //BatchedSubgraphLayer0b(const BatchedSubgraphLayer0b& x, const int _dev):
    //BatchedSubgraphLayer0b(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    static BatchedSubgraphLayer0b linmaps(const SOURCE& x){
      BatchedSubgraphLayer0b R(x.G,x.S,x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    BatchedSubgraphLayer0b(const SOURCE& x, const Subgraph& _S, const int min_overlaps=1):
      BatchedSubgraphLayer0b(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x,min_overlaps);
    }

    template<typename SOURCE>
    BatchedSubgraphLayer0b(const SOURCE& x, const BatchedGgraph& _G, const Subgraph& _S, const int min_overlaps=1):
      BatchedSubgraphLayer0b(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x,min_overlaps);
    }

  };




}

#endif 


