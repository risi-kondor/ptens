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

#ifndef _ptens_SubgraphLayer1
#define _ptens_SubgraphLayer1

#include "Ggraph.hpp"
#include "Subgraph.hpp"
#include "SubgraphLayer0.hpp"
#include "Ptensors1.hpp"
//#include "SubgraphLayer.hpp"


namespace ptens{


  template<typename TYPE> 
  class SubgraphLayer1: public Ptensors1<TYPE>{
  public:

    typedef Ptensors1<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    //typedef cnine::Ltensor<TYPE> TENSOR;

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


    SubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const BASE& x):
      BASE(x), G(_G), S(_S){}

    SubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const AtomsPack& atoms, const TENSOR& x):
      BASE(atoms,x), G(_G), S(_S){}

    SubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_G.subgraphs(_S),nc,fcode,_dev){}

    SubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev=0):
      G(_G), S(_S), BASE(_atoms,nc,fcode,_dev){}

    SubgraphLayer1(const SubgraphLayer1& x, const int _dev):
      SubgraphLayer1(x.G,x.S,BASE(x,_dev)){}

    //static SubgraphLayer1 cat(const vector<SubgraphLayer1>& list){
    //vector<AtomsPack1> v;
    //for(auto& p:list)
    //v.push_back(p.atoms);
    //return SubgraphLayer1(AtomsPack1::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    //}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    SubgraphLayer1 copy() const{
      return SubgraphLayer1(G,S,BASE::copy());
    }

    SubgraphLayer1 copy(const int _dev) const{
      return SubgraphLayer1(G,S,BASE::copy(_dev));
    }

    SubgraphLayer1 zeros_like() const{
      return SubgraphLayer1(G,S,BASE::zeros_like());
    }

    SubgraphLayer1 zeros_like(const int nc) const{
      return SubgraphLayer1(G,S,BASE::zeros_like(nc));
    }

    SubgraphLayer1 gaussian_like() const{
      return SubgraphLayer1(G,S,BASE::gaussian_like());
    }

    //static SubgraphLayer1 like(const SubgraphLayer1& x, const cnine::Ltensor<TYPE>& M){
    //return SubgraphLayer1(x.G,x.S);
    //}

    static SubgraphLayer1* new_zeros_like(const SubgraphLayer1& x){
      return new SubgraphLayer1(x.zeros_like());
    }


  public: // ---- Access -------------------------------------------------------------------------------------

  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    template<typename SOURCE>
    SubgraphLayer1(const SOURCE& x, const Subgraph& _S)://, const int min_overlaps=1):
      SubgraphLayer1(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev){
      add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
    }

    template<typename SOURCE>
    SubgraphLayer1(const SOURCE& x, const Ggraph& _G, const Subgraph& _S)://, const int min_overlaps=1):
      SubgraphLayer1(_G,_S,_G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev){
      add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static SubgraphLayer1 linmaps(const SOURCE& x){
      SubgraphLayer1 R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }


  public: // ---- Autobahn -----------------------------------------------------------------------------------


    SubgraphLayer1 schur(const TENSOR& W, const TENSOR& B) const{
      SubgraphLayer1 R=zeros_like(W.dims[2]);
      R.add_schur(W,B);
      return R;
    }


    void add_schur(const SubgraphLayer1& x, const TENSOR& W, const TENSOR& B) const{
      auto& S=x.S;
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==x.get_nc());
      PTENS_ASSRT(W.dims[2]==get_nc());
      PTENS_ASSRT(B.dims.size()==2);
      PTENS_ASSRT(B.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(B.dims[1]==W.dims[2]);

      x.for_each_eigenslice(view3(K),x.view3(K),[&]
	(cnine::Rtensor2_view rslice, cnine::Rtensor2_view xslice, const int b){
	  rslice.add_matmul_AA(xslice,W.view3().slice0(b)); // OK
	  rslice.add_broadcast0(B.view2().slice0(b)); // OK	
	},true);
    }


   void add_schur_back0(const Ptensors1<TYPE>& r, const TENSOR& W){
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(W.dims[2]==r.get_nc());

      for_each_eigenslice(view3(K),r.view3(K),[&]
	(cnine::Rtensor2_view rslice, cnine::Rtensor2_view xslice, const int b){
	  rslice.add_matmul_AT(xslice,W.view3().slice0(b)); // OK
	},true);
    }


    void add_schur_back1_to(const TENSOR& W, const TENSOR& B, const Ptensors1<TYPE>& r){
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(W.dims[2]==r.get_nc());
      PTENS_ASSRT(B.dims.size()==2);
      PTENS_ASSRT(B.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(B.dims[1]==r.get_nc());

      for_each_eigenslice(view3(K),r.view3(K),[&]
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

      auto X=TENSOR({N,K,xnc},0,x.dev);
      if(!inplace_add) X.view3().add_mprod(E.transp(),x);

      auto Y=TENSOR({N,K,ync},0,x.dev);
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
  inline SubgraphLayer1<float> sglinmaps1(const SOURCE& x){
    SubgraphLayer1<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer1<float> gather1(const SOURCE& x, const Subgraph& _S){
    SubgraphLayer1<float> R(x.G,_S,x.G.subgraphs(_S),x.get_nc()*vector<int>({1,2,5})[x.getk()],0,x.dev);
    R.add_gather(x,LayerMap::overlaps_map(R.atoms,x.atoms));
    return R;
  }


}

#endif 

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

