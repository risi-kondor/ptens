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

#ifndef _ptens_BatchedSubgraphLayer0
#define _ptens_BatchedSubgraphLayer0

#include "BatchedGgraph.hpp"
#include "Subgraph.hpp"
#include "BatchedPtensors0.hpp"
//#include "Ptensors1b.hpp"
//#include "Ptensors2b.hpp"
//#include "BatchedSubgraphLayerb.hpp"


namespace ptens{


  template<typename TYPE> 
  class BatchedSubgraphLayer0: public BatchedPtensors0<TYPE>{
  public:

    typedef BatchedPtensors0<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    //typedef BatchedAtomsPackN<AtomsPack0obj<int> > BatchedAtomsPack0;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;

    const BatchedGgraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedSubgraphLayer0(const BatchedGgraph& _G, const TENSOR& x):
    //BASE(x), G(_G), S(Subgraph::trivial()){}

    BatchedSubgraphLayer0(const BatchedGgraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    BatchedSubgraphLayer0(const BatchedGgraph& _G, const Subgraph& _S, const TENSOR& M):
      BASE(_G.subgraphs(_S),M), G(_G), S(_S){}

    BatchedSubgraphLayer0(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    //BatchedSubgraphLayer0(const BatchedGgraph& _G, const int nc, const int fcode=0, const int _dev=0):
    //G(_G), S(Subgraph::trivial()), BASE(_G.getn(),nc,fcode,_dev){}

    BatchedSubgraphLayer0(const BatchedGgraph& _G, const Subgraph& _S, const BatchedAtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    /*
    static BatchedSubgraphLayer0 cat(const vector<BatchedSubgraphLayer0>& list){
      vector<AtomsPack0> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return BatchedSubgraphLayer0(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }
    */

    static BatchedSubgraphLayer0 from_vertex_features(const vector<int>& graphs, const TENSOR& M){
      BatchedGgraph G(graphs);
      //vector<int> sizes;
      //for(int i=0; i<G.size(); i++)
      //sizes.push_back(G[i].getn());
      return BatchedSubgraphLayer0(G,Subgraph::trivial(),M);
    }

    /*
    static BatchedSubgraphLayer0 from_edge_features(const vector<int>& graphs, const TENSOR& M){
      BatchedGgraph G(graphs);
      auto atoms=new BatchedAtomsPackObj();
      for(int i=0; i<G.size(); i++)
	atoms->push_back(to_share(new AtomsPackObj(G[i].original_edges())));
      atoms->make_row_offsets();
      return BatchedSubgraphLayer0(G,Subgraph::edge(),BASE(BatchedAtomsPack0(atoms),M));
    }
    */

  public: // ----- Spawning ----------------------------------------------------------------------------------


    BatchedSubgraphLayer0 copy() const{
      return BatchedSubgraphLayer0(G,S,BASE::copy());
    }

    BatchedSubgraphLayer0 copy(const int _dev) const{
      return BatchedSubgraphLayer0(G,S,BASE::copy(_dev));
    }

    BatchedSubgraphLayer0 zeros_like() const{
      return BatchedSubgraphLayer0(G,S,BASE::zeros_like());
    }

    BatchedSubgraphLayer0 gaussian_like() const{
      return BatchedSubgraphLayer0(G,S,BASE::gaussian_like());
    }

    static BatchedSubgraphLayer0* new_zeros_like(const BatchedSubgraphLayer0& x){
      return new BatchedSubgraphLayer0(x.G,x.S,x.BASE::zeros_like());
    }
    
    //BatchedSubgraphLayer0(const BatchedSubgraphLayer0& x, const int _dev):
    //BatchedSubgraphLayer0(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    static BatchedSubgraphLayer0 linmaps(const SOURCE& x){
      BatchedSubgraphLayer0 R(x.G,x.S,x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    BatchedSubgraphLayer0(const SOURCE& x, const Subgraph& _S, const int min_overlaps=1):
      BatchedSubgraphLayer0(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x,min_overlaps);
    }

    template<typename SOURCE>
    BatchedSubgraphLayer0(const SOURCE& x, const BatchedGgraph& _G, const Subgraph& _S, const int min_overlaps=1):
      BatchedSubgraphLayer0(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,1,2})[x.getk()],0,x.dev){
      add_gather(x,min_overlaps);
    }

  };




}

#endif 


