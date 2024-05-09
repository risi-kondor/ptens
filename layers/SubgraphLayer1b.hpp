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
//#include "Rtensor3_view.hpp"


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
    using BASE::size;
    using BASE::get_nc;
    using BASE::get_dev;
    using BASE::get_grad;
    using BASE::view3;
    using BASE::cols;

    using TENSOR::dim;
    using TENSOR::get_arr;

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

    SubgraphLayer1b zeros_like(const int nc) const{
      return SubgraphLayer1b(G,S,BASE::zeros_like(nc));
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


  public: // ---- Access -------------------------------------------------------------------------------------

    /*
    const cnine::Rtensor3_view view3() const{
      int K=S.getn();
      int nc=get_nc();
      return cnine::Rtensor3_view(const_cast<float*>(get_arr()),dim(0)/K,K,nc,K*nc,nc,1);
    }

    cnine::Rtensor3_view view3(){
      int K=S.getn();
      int nc=get_nc();
      return cnine::Rtensor3_view(get_arr(),dim(0)/K,K,nc,K*nc,nc,1);
    }
    */

  public: // ---- Message passing between subgraph layers -----------------------------------------------------


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


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static SubgraphLayer1b linmaps(const SOURCE& x){
      SubgraphLayer1b R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors1b<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      cols(nc,nc)+=x;
    }

    void add_linmaps_back(const Ptensors1b<TYPE>& r){
      int nc=get_nc();
      broadcast0(r.reduce0(0,nc));
      add(r.cols(nc,nc));
    }

    Ptensorsb<TYPE> reduce0() const{
      TimedFn T("SubgraphLayer1b","reduce0",*this);
      Ptensorsb<TYPE> R({size(),get_nc()},0,get_dev());
      view3(S.getn()).sum1_into(R.view2());
      return R;
    }

    Ptensorsb<TYPE> reduce0(const int offs, const int nc) const{
      TimedFn T("SubgraphLayer1b","reduce0",*this);
      Ptensorsb<TYPE> R({size(),nc},0,get_dev());
      view3(S.getn(),offs,nc).sum1_into(R.view2());
      return R;
    }

    void broadcast0(const Ptensorsb<TYPE>& x, const int offs=0){
      TimedFn T("SubgraphLayer1b","broadcast0",*this);
      PTENS_ASSRT(x.ndims()==2);
      view3(S.getn(),offs,x.dim(1))+=cnine::repeat1(x.view2(),S.getn());
    }


  public: // ---- Autobahn -----------------------------------------------------------------------------------


    SubgraphLayer1b autobahn(const TENSOR& W, const TENSOR& B) const{
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(B.dims.size()==2);
      PTENS_ASSRT(B.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(B.dims[1]==W.dims[2]);

      SubgraphLayer1b R=zeros_like(W.dims[2]);
      for_each_eigenslice(R.view3(K),view3(K),[&]
	(cnine::Rtensor2_view rslice, cnine::Rtensor2_view xslice, const int b){
	  rslice.add_matmul_AA(xslice,W.view3().slice0(b)); // OK
	  rslice.add_broadcast0(B.view2().slice0(b)); // OK	
	},true);
      return R;
    }


   void add_autobahn_back0(const Ptensors1b<TYPE>& r, const TENSOR& W){
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(W.dims[2]==r.get_nc());

      for_each_eigenslice(get_grad().view3(K),r.get_grad().view3(K),[&]
	(cnine::Rtensor2_view rslice, cnine::Rtensor2_view xslice, const int b){
	  rslice.add_matmul_AT(xslice,W.view3().slice0(b)); // OK
	},true);
    }


    void add_autobahn_back1_to(const TENSOR& W, const TENSOR& B, const Ptensors1b<TYPE>& r){
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(W.dims[2]==r.get_nc());
      PTENS_ASSRT(B.dims.size()==2);
      PTENS_ASSRT(B.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(B.dims[1]==r.get_nc());

      for_each_eigenslice(view3(K),r.get_grad().view3(K),[&]
	(cnine::Rtensor2_view xslice, cnine::Rtensor2_view rslice, const int b){
	  W.view3().slice0(b).add_matmul_TA(xslice,rslice); // OK
	  rslice.sum0_into(B.view2().slice0(b)); 
	});
    }


    void for_each_eigenslice(const cnine::Rtensor3_view x, const cnine::Rtensor3_view y,
      std::function<void(cnine::Rtensor2_view xslice, cnine::Rtensor2_view yslice, const int b)> lambda,
			     const bool inplace_add=false) const{
      S.make_eigenbasis();
      int N=x.n0;
      int K=x.n1;
      int xnc=x.n2;
      int ync=y.n2;
      int nblocks=S.obj->eblocks.size();

      S.obj->evecs.move_to_device(x.dev);
      cnine::Rtensor2_view E=S.obj->evecs.view2();
      const auto& blocks=S.obj->eblocks;

      PTENS_ASSRT(y.n0==N);
      PTENS_ASSRT(y.n1==K);
      PTENS_ASSRT(E.n0==K);
      PTENS_ASSRT(E.n1==K);
      PTENS_ASSRT(x.dev==y.dev);

      auto X=cnine::Ltensor<float>({N,K,xnc},0,x.dev);
      if(!inplace_add) X.view3().add_mprod(E.transp(),x);

      auto Y=cnine::Tensor<float>({N,K,ync},0,x.dev);
      Y.view3().add_mprod(E.transp(),y);

      int offs=0;
      for(int b=0; b<nblocks; b++){
	for(int i=offs; i<offs+blocks[b]; i++){
	  lambda(X.view3().slice1(i),Y.view3().slice1(i),b);
	}
	offs+=blocks[b];
      }

      if(inplace_add) const_cast<cnine::Rtensor3_view&>(x).add_mprod(E,X.view3());
    }


   public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Subgraphlayer1b";
    }


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

