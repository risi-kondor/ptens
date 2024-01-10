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

#ifndef _ptens_SubgraphLayer2b
#define _ptens_SubgraphLayer2b

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "SubgraphLayer1b.hpp"
#include "Ptensors2b.hpp"


namespace ptens{

  //template<typename TYPE> class SubgraphLayer1b;
  //template<typename TYPE> class SubgraphLayer2b;
  //template<typename TYPE> inline SubgraphLayer1b<TYPE> gather(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S);


  template<typename TYPE> 
  class SubgraphLayer2b: public Ptensors2b<TYPE>{
  public:

    typedef Ptensors2b<TYPE> BASE;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::add_gather;
    //using BASE::G;
    //using BASE::S;
    //using TLAYER::dev;
    //using TLAYER::getn;
    //using TLAYER::get_nc;
    //using TLAYER::get_grad;
    //using TLAYER::tensor;
    //using TLAYER::inp;
    //using TLAYER::diff2;

    const Ggraph G;
    const Subgraph S;


  public: // ----- Constructors ------------------------------------------------------------------------------


    SubgraphLayer2b(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer2b(const Ggraph& _G, const Subgraph& _S, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_G.subgraphs(_S),nc,fcode,_dev){}

    SubgraphLayer2b(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    static SubgraphLayer2b cat(const vector<SubgraphLayer2b>& list){
      vector<AtomsPack2> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return SubgraphLayer2b(AtomsPack2::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }


  public: // ----- Spawning ----------------------------------------------------------------------------------


    SubgraphLayer2b copy() const{
      return SubgraphLayer2b(G,S,BASE::copy());
    }

    SubgraphLayer2b copy(const int _dev) const{
      return SubgraphLayer2b(G,S,BASE::copy(_dev));
    }

    SubgraphLayer2b zeros_like() const{
      return SubgraphLayer2b(G,S,BASE::zeros_like());
    }

    SubgraphLayer2b gaussian_like() const{
      return SubgraphLayer2b(G,S,BASE::gaussian_like());
    }

    static SubgraphLayer2b* new_zeros_like(const SubgraphLayer2b& x){
      return new SubgraphLayer2b(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    SubgraphLayer2b(const SubgraphLayer2b& x, const int _dev):
      SubgraphLayer2b(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    //SubgraphLayer1b(const NodeLayerb<TYPE>& x, const Subgraph& _S):
    //SubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),x.get_dev()){
    //add_gather(x);
    //}

    //void gather_back(NodeLayer& x){
    //x.get_grad().emp_fromB(get_grad());
    //}


    SubgraphLayer2b(const SubgraphLayer0b<float>& x, const Subgraph& _S):
      SubgraphLayer2b(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),0,x.dev){
      add_gather(x);
    }

    SubgraphLayer2b(const SubgraphLayer1b<float>& x, const Subgraph& _S):
      SubgraphLayer2b(x.G,_S,x.G.subgraphs(_S),5*x.get_nc(),0,x.dev){
      add_gather(x);
    }

    SubgraphLayer2b(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S):
      SubgraphLayer2b(x.G,_S,x.G.subgraphs(_S),15*x.get_nc(),0,x.dev){
      add_gather(x);
    }

  };



  template<typename SOURCE>
  inline SubgraphLayer2b<float> sglinmaps2(const SOURCE& x){
    SubgraphLayer2b<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer2b<float> gather2(const SOURCE& x, const Subgraph& _S){
    SubgraphLayer2b<float> R(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({2,5,15})[x.getk()],0,x.dev);
    R.add_gather(x);
    return R;
  }



}

#endif 
  /*
  template<typename TYPE>
  inline SubgraphLayer2b<TYPE> gather2(const SubgraphLayer0b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer2b<TYPE> R(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }

  template<typename TYPE>
  inline SubgraphLayer2b<TYPE> gather2(const SubgraphLayer1b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer2b<TYPE> R(x.G,_S,x.G.subgraphs(_S),5*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }

  template<typename TYPE>
  inline SubgraphLayer2b<TYPE> gather2(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer2b<TYPE> R(x.G,_S,x.G.subgraphs(_S),15*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }
  */
