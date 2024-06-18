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

#include "SubgraphLayer0b.hpp"


namespace ptens{

  template<typename TLAYER> 
  class SubgraphLayer0;
  template<typename TLAYER> 
  class SubgraphLayer2;

  template<typename TLAYER> 
  class SubgraphLayer1: public SubgraphLayer<TLAYER>{
  public:

    typedef cnine::RtensorA rtensor;
    typedef SubgraphLayer<TLAYER> BASE;

    using BASE::BASE;
    using BASE::atoms;
    using BASE::G;
    using BASE::S;
    using TLAYER::dev;
    using TLAYER::getn;
    using TLAYER::get_nc;
    using TLAYER::get_grad;
    using TLAYER::inp;
    using TLAYER::diff2;
    using TLAYER::view3;
    using TLAYER::overlaps;


  public: // ---- Named Constructors ------------------------------------------------------------------------------------------


    static SubgraphLayer1<TLAYER> zeros_like(const SubgraphLayer1<TLAYER>& x){
      return SubgraphLayer1(TLAYER::zeros_like(x),x.G,x.S);
    }

    static SubgraphLayer1<TLAYER> randn_like(const SubgraphLayer1<TLAYER>& x){
      return SubgraphLayer1(TLAYER::randn_like(x),x.G,x.S);
    }

    SubgraphLayer1<TLAYER> zeros() const{
      return SubgraphLayer1(TLAYER::zeros_like(*this),G,S);
    }

    SubgraphLayer1<TLAYER> zeros(const int _nc) const{
      return SubgraphLayer1(TLAYER::zeros_like(*this,_nc),G,S);
    }

    static SubgraphLayer1<TLAYER> like(const SubgraphLayer1& x, const cnine::RtensorA& M){
      return SubgraphLayer1(TLAYER::like(x,M),x.G,x.S);
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    SubgraphLayer1(const SubgraphLayer1<TLAYER>& x, const int _dev):
      SubgraphLayer<TLAYER>(TLAYER(x,_dev),x.G,x.S){}


  public: // ---- Access --------------------------------------------------------------------------------------


    int n_eblocks() const{
      return S.n_eblocks();
    }


  public: // ---- Message passing between subgraph layers -----------------------------------------------------


    SubgraphLayer1(const NodeLayer& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),x.dev){
	x.emp_to(*this);
    }

    void gather_back(NodeLayer& x){
      x.get_grad().emp_fromB(get_grad());
    }


    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,x.G.subgraphs(_S),x.get_nc(),x.dev){
      emp01(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer0<TLAYER2>& x){
      emp10(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,x.G.subgraphs(_S),2*x.get_nc(),x.dev){
      emp11(*this,x,overlaps(x.atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer1<TLAYER2>& x){
      emp11_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer2<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,x.G.subgraphs(_S),5*x.get_nc(),x.dev){
      emp21(*this,x,overlaps(x.atoms)); 
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer2<TLAYER2>& x){
      emp21_back(x.get_grad(),get_grad(),x.overlaps(atoms));
    }


  public: // ---- Message passing from Ptensor layers ---------------------------------------------------------


    SubgraphLayer1(const Ptensors0& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1(_G,_S,_G.subgraphs(_S),x.get_nc(),x.dev){
      emp01(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors0& x){
      emp10(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    SubgraphLayer1(const Ptensors1& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1(_G,_S,_G.subgraphs(_S),2*x.get_nc(),x.dev){
      cnine::ftimer timer("SubgraphLayer1 from Ptensors1");
      emp11(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors1& x){
      emp11_back(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }

    SubgraphLayer1(const Ptensors2& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1(_G,_S,_G.subgraphs(_S),5*x.get_nc(),x.dev){
      emp21(*this,x,overlaps(x.atoms));
    }

    void gather_back(Ptensors2& x){
      emp21_back(x.get_grad(),get_grad(),x.overlaps(atoms)); 
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    SubgraphLayer1 permute(const cnine::permutation& pi){
      return SubgraphLayer1(Ptensors1::permute(pi),G.permute(pi),S);
    }


  public: // ---- Autobahn -----------------------------------------------------------------------------------


    SubgraphLayer1<TLAYER> autobahn(const cnine::RtensorA& W, const cnine::RtensorA& B) const{
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(B.dims.size()==2);
      PTENS_ASSRT(B.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(B.dims[1]==W.dims[2]);
      //add_autobahn(R,*this,S.obj->evecs.view2(),S.obj->eblocks,W.view3(),B.view2());

      SubgraphLayer1<TLAYER> R(TLAYER::zeros_like(*this,W.dims[2]),G,S);
      add_to_each_eigenslice(R.view3(K),view3(K),[&]
	(cnine::Rtensor2_view rslice, cnine::Rtensor2_view xslice, const int b){
	  rslice.add_matmul_AA(xslice,W.view3().slice0(b)); // OK
	  rslice.add_broadcast0(B.view2().slice0(b)); // OK	
	});
      return R;
    }

    
    void add_autobahn_back0(SubgraphLayer1<TLAYER>& r, const cnine::RtensorA& W){
      S.make_eigenbasis();
      int K=S.getn();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(W.dims[2]==r.get_nc());
      //cnine::Rtensor3_view Wt(W.mem(),W.dims[0],W.dims[2],W.dims[1],W.strides[0],W.strides[2],W.strides[1],W.dev);
      //add_autobahn(get_grad(),r.get_grad(),S.obj->evecs.view2(),S.obj->eblocks,Wt);

      add_to_each_eigenslice(get_grad().view3(K),r.get_grad().view3(K),[&]
	(cnine::Rtensor2_view rslice, cnine::Rtensor2_view xslice, const int b){
	  rslice.add_matmul_AT(xslice,W.view3().slice0(b)); // OK
	});
    }


    void add_autobahn_back1_to(const cnine::RtensorA& W, const cnine::RtensorA& B, SubgraphLayer1<TLAYER>& r){
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
      std::function<void(const cnine::Rtensor2_view& xslice, const cnine::Rtensor2_view& yslice, const int b)> lambda) const{
      S.make_eigenbasis();
      int N=x.n0;
      int K=x.n1;
      int xnc=x.n2;
      int ync=y.n2;
      int nblocks=S.obj->eblocks.size();

      cnine::Rtensor2_view E=S.obj->evecs.view2();
      const auto& blocks=S.obj->eblocks;

      PTENS_ASSRT(y.n0==N);
      PTENS_ASSRT(y.n1==K);
      PTENS_ASSRT(E.n0==K);
      PTENS_ASSRT(E.n1==K);
      PTENS_ASSRT(x.dev==y.dev);

      auto X=cnine::Tensor<float>::zero({N,K,xnc},x.dev);
      X.view3().add_mprod(E.transp(),x);

      auto Y=cnine::Tensor<float>::zero({N,K,ync},x.dev);
      Y.view3().add_mprod(E.transp(),y);

      int offs=0;
      for(int b=0; b<nblocks; b++){
	for(int i=offs; i<offs+blocks[b]; i++)
	  lambda(X.view3().slice1(i),Y.view3().slice1(i),b);
	offs+=blocks[b];
      }
    }


    void add_to_each_eigenslice(cnine::Rtensor3_view r, const cnine::Rtensor3_view x,
      std::function<void(const cnine::Rtensor2_view& rslice, const cnine::Rtensor2_view& xslice, const int b)> lambda) const{
      S.make_eigenbasis();
      int N=r.n0;
      int K=r.n1;
      int nc=r.n2;
      int xnc=x.n2;
      int nblocks=S.obj->eblocks.size();

      cnine::Rtensor2_view E=S.obj->evecs.view2();
      const auto& blocks=S.obj->eblocks;

      PTENS_ASSRT(x.n0==N);
      PTENS_ASSRT(x.n1==K);
      PTENS_ASSRT(E.n0==K);
      PTENS_ASSRT(E.n1==K);
      PTENS_ASSRT(r.dev==x.dev);

      auto A=cnine::Tensor<float>::zero({N,K,xnc},x.dev);
      A.view3().add_mprod(E.transp(),x);

      auto B=cnine::Tensor<float>::zero({N,K,nc},x.dev);
      int offs=0;
      for(int b=0; b<nblocks; b++){
	for(int i=offs; i<offs+blocks[b]; i++)
	  lambda(B.view3().slice1(i),A.view3().slice1(i),b);
	offs+=blocks[b];
      }

      r.add_mprod(E,B.view3());
    }





  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SubgraphLayer1";
    }

    string repr() const{
      if(dev==0) return "<SubgraphLayer1[N="+to_string(getn())+"]>";
      else return "<SubgraphLayer1[N="+to_string(getn())+"][G]>";
    }




  };

}

#endif 
