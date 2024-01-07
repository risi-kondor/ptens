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

#ifndef _ptens_SubgraphLayer0b
#define _ptens_SubgraphLayer0b

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "Ptensors0b.hpp"
#include "SubgraphLayerb.hpp"


namespace ptens{

  //template<typename TYPE> class SubgraphLayer0b;
  //template<typename TYPE> class SubgraphLayer1b;
  //template<typename TYPE> class SubgraphLayer2b;

  //template<typename TYPE> inline SubgraphLayer0b<TYPE> gather0(const SubgraphLayer1b<TYPE>& x, const Subgraph& _S);
  //template<typename TYPE> inline SubgraphLayer0b<TYPE> gather0(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S);


  template<typename TYPE> 
  class SubgraphLayer0b: public Ptensors0b<TYPE>{
  public:

    typedef Ptensors0b<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

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


    SubgraphLayer0b(const Ggraph& _G, const TENSOR& x):
      BASE(x), G(_G), S(Subgraph::trivial()){}

    SubgraphLayer0b(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer0b(const Ggraph& _G, const int nc, const int fcode=0, const int _dev=0):
      G(_G), S(Subgraph::trivial()), BASE(_G.getn(),nc,fcode,_dev){}

    SubgraphLayer0b(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    SubgraphLayer0b copy() const{
      return SubgraphLayer0b(G,S,BASE::copy());
    }

    SubgraphLayer0b copy(const int _dev) const{
      return SubgraphLayer0b(G,S,BASE::copy(_dev));
    }

    SubgraphLayer0b zeros_like() const{
      return SubgraphLayer0b(G,S,BASE::zeros_like());
    }

    SubgraphLayer0b gaussian_like() const{
      return SubgraphLayer0b(G,S,BASE::gaussian_like());
    }

    static SubgraphLayer0b* new_zeros_like(const SubgraphLayer0b& x){
      return new SubgraphLayer0b(x.G,x.S,x.TENSOR::zeros_like());
    }
    
    SubgraphLayer0b(const SubgraphLayer0b& x, const int _dev):
      SubgraphLayer0b(x.G,x.S,BASE(x,_dev)){}


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    //SubgraphLayer0b(const SubgraphLayer0b<float>& x, const Subgraph& _S):
    //SubgraphLayer0b(x.G,_S,x.G.subgraphs(_S),x.get_nc(),0,x.dev){
    //add_gather(x);
    //}

    //SubgraphLayer0b(const SubgraphLayer1b<float>& x, const Subgraph& _S):
    //SubgraphLayer0b(gather0<TYPE>(x,_S)){}

    //SubgraphLayer0b(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S):
    //SubgraphLayer0b(gather0<TYPE>(x,_S)){}


  };


  template<typename SOURCE>
  inline SubgraphLayer0b<float> sglinmaps0(const SOURCE& x){
    SubgraphLayer0b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer0b<float> gather0(const SOURCE& x, const Subgraph& _S){
    int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
    SubgraphLayer0b<float> R(x.G,_S,x.G.subgraphs(_S),nc,0,x.dev);
    R.add_gather(x);
    return R;
  }


}

#endif 


  /*
  template<typename TYPE>
  inline SubgraphLayer0b<TYPE> gather0(const SubgraphLayer1b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer0b<TYPE> R(x.G,_S,x.G.subgraphs(_S),x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }

  template<typename TYPE>
  inline SubgraphLayer0b<TYPE> gather0(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer0b<TYPE> R(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }
  */
