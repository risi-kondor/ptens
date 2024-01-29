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

#ifndef _ptens_SubgraphLayer1b
#define _ptens_SubgraphLayer1b

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "SubgraphLayer0b.hpp"
#include "Ptensors1b.hpp"
#include "SubgraphLayerb.hpp"


namespace ptens{

  //template<typename TYPE> class SubgraphLayer1b;
  //template<typename TYPE> class SubgraphLayer2b;

  //template<typename TYPE> inline SubgraphLayer1b<TYPE> gather1(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S);


  template<typename TYPE> 
  class SubgraphLayer1b: public Ptensors1b<TYPE>{
  public:

    typedef Ptensors1b<TYPE> BASE;
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


    SubgraphLayer1b(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer1b(const Ggraph& _G, const Subgraph& _S, const AtomsPack1& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    SubgraphLayer1b(const Ggraph& _G, const Subgraph& _S, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_G.subgraphs(_S),nc,fcode,_dev){}

    SubgraphLayer1b(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    SubgraphLayer1b(const SubgraphLayer1b& x, const int _dev):
      SubgraphLayer1b(x.G,x.S,BASE(x,_dev)){}

    static SubgraphLayer1b cat(const vector<SubgraphLayer1b>& list){
      vector<AtomsPack1> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return SubgraphLayer1b(AtomsPack1::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }

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

    //static SubgraphLayer1b like(const SubgraphLayer1b& x, const cnine::Ltensor<TYPE>& M){
    //return SubgraphLayer1(x.G,x.S);
    //}

    static SubgraphLayer1b* new_zeros_like(const SubgraphLayer1b& x){
      return new SubgraphLayer1b(x.zeros_like());
    }
    

  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    static SubgraphLayer1b linmaps(const SOURCE& x){
      SubgraphLayer1b R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    SubgraphLayer1b(const SOURCE& x, const Subgraph& _S):
      SubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev){
      add_gather(x);
    }

    template<typename SOURCE>
    SubgraphLayer1b(const SOURCE& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1b(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev){
      add_gather(x);
    }

    //SubgraphLayer1b(const Ptensors0b<TYPE>& x, const Ggraph& g, const Subgraph& s):
    //SubgraphLayer1b(g,s,g.subgraphs(s),x.get_nc(),0,x.dev){
    //add_gather(x);
    //}

    //SubgraphLayer1b(const Ptensors1b<TYPE>& x, const Ggraph& g, const Subgraph& s):
    //SubgraphLayer1b(g,s,g.subgraphs(s),2*x.get_nc(),0,x.dev){
      //add_gather(x);
    //}

    //SubgraphLayer1b(const Ptensors2b<TYPE>& x, const Ggraph& g, const Subgraph& s):
    //SubgraphLayer1b(g,s,g.subgraphs(s),5*x.get_nc(),0,x.dev){
    //add_gather(x);
    ///}

    //SubgraphLayer1b(const NodeLayerb<TYPE>& x, const Subgraph& _S):
    //SubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),x.get_dev()){
    //add_gather(x);
    //}

    //void gather_back(NodeLayer& x){
    //x.get_grad().emp_fromB(get_grad());
    //}



  };


  template<typename SOURCE>
  inline SubgraphLayer1b<float> sglinmaps1(const SOURCE& x){
    SubgraphLayer1b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer1b<float> gather1(const SOURCE& x, const Subgraph& _S){
    SubgraphLayer1b<float> R(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev);
    R.add_gather(x);
    return R;
  }


}

#endif 
  /*
  template<typename TYPE>
  inline SubgraphLayer1b<TYPE> gather1(const SubgraphLayer0b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer1b<TYPE> R(x.G,_S,x.G.subgraphs(_S),1*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }

  template<typename TYPE>
  inline SubgraphLayer1b<TYPE> gather1(const SubgraphLayer1b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer1b<TYPE> R(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }

  template<typename TYPE>
  inline SubgraphLayer1b<TYPE> gather1(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S){
    SubgraphLayer1b<TYPE> R(x.G,_S,x.G.subgraphs(_S),5*x.get_nc(),0,x.dev);
    R.add_gather(x);
    return R;
  }
  */
    /*
    SubgraphLayer1b(const SubgraphLayer0b<float>& x, const Subgraph& _S):
      SubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),x.get_nc(),0,x.dev){
      add_gather(x);
    }

    SubgraphLayer1b(const SubgraphLayer1b<float>& x, const Subgraph& _S):
      SubgraphLayer1b(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),0,x.dev){
      add_gather(x);
    }

    SubgraphLayer1b(const SubgraphLayer2b<TYPE>& x, const Subgraph& _S):
      SubgraphLayer1b(gather1<TYPE>(x,_S)){}
    */
    //template<typename SOURCE>
    //inline SubgraphLayer1b<float> gather(const SOURCE& x, const Subgraph& _S){
    //SubgraphLayer1b<float> R(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev);
    //R.add_gather(x);
    //return R;
    //}

