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

#include "Hgraph.hpp"
#include "Subgraph.hpp"
#include "FindPlantedSubgraphs.hpp"
#include "TransferMap.hpp"
#include "EMPlayers2.hpp"
#include "SubgraphLayer0.hpp"


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


  public: 

    //template<typename IPACK>
    //SubgraphLayer1(const Ggraph& _G, const Subgraph& _S, const IPACK& ipack, const int nc, const int _dev=0):
    //G(_G), S(_S), TLAYER(ipack,nc,cnine::fill_zero(),_dev){}


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


  public: // ---- Transport ----------------------------------------------------------------------------------


    SubgraphLayer1(const SubgraphLayer1<TLAYER>& x, const int _dev):
      SubgraphLayer<TLAYER>(TLAYER(x,_dev),x.G,x.S){}


  public: // ---- Message passing ----------------------------------------------------------------------------------------


    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer0<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,CachedPlantedSubgraphsMx(*x.G.obj,*_S.obj),x.get_nc(),x.dev){
      emp01(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer0<TLAYER2>& x){
      emp10(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer1<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,CachedPlantedSubgraphsMx(*x.G.obj,*_S.obj),2*x.get_nc(),x.dev){
      emp11(*this,x,TransferMap(x.atoms,atoms));
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer1<TLAYER2>& x){
      emp11_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms));
    }

    template<typename TLAYER2>
    SubgraphLayer1(const SubgraphLayer2<TLAYER2>& x, const Subgraph& _S):
      SubgraphLayer1(x.G,_S,CachedPlantedSubgraphsMx(*x.G.obj,*_S.obj),5*x.get_nc(),x.dev){
      emp21(*this,x,TransferMap(x.atoms,atoms)); 
    }

    template<typename TLAYER2>
    void gather_back(SubgraphLayer2<TLAYER2>& x){
      emp21_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms));
    }


    SubgraphLayer1(const Ptensors0& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1(_G,_S,CachedPlantedSubgraphsMx(*_G.obj,*_S.obj),x.get_nc(),x.dev){
      emp01(*this,x,TransferMap(x.atoms,atoms));
    }

    void gather_back(Ptensors0& x){
      emp10(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms)); 
    }

    SubgraphLayer1(const Ptensors1& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1(_G,_S,CachedPlantedSubgraphsMx(*_G.obj,*_S.obj),2*x.get_nc(),x.dev){
      emp11(*this,x,TransferMap(x.atoms,atoms));
    }

    void gather_back(Ptensors1& x){
      emp11_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms)); 
    }

    SubgraphLayer1(const Ptensors2& x, const Ggraph& _G, const Subgraph& _S):
      SubgraphLayer1(_G,_S,CachedPlantedSubgraphsMx(*_G.obj,*_S.obj),5*x.get_nc(),x.dev){
      emp21(*this,x,TransferMap(x.atoms,atoms));
    }

    void gather_back(Ptensors2& x){
      emp21_back(x.get_grad(),get_grad(),TransferMap(atoms,x.atoms)); 
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    SubgraphLayer1 permute(const cnine::permutation& pi){
      return SubgraphLayer1(G.permute(pi),S,Ptensors1::permute(pi));
    }


  public: // ---- Autobahn -----------------------------------------------------------------------------------


    SubgraphLayer1<TLAYER> autobahn(const cnine::Tensor<float>& W) const{
      S.make_eigenbasis();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      SubgraphLayer1<TLAYER> R(TLAYER::zeros_like(*this,W.dims[2]),G,S);
      add_autobahn(R,*this,S.obj->evecs.view2(),S.obj->eblocks,W.view3());
      return R;
    }
    
    void add_autobahn_back0(SubgraphLayer1<TLAYER>& r, const cnine::Tensor<float>& W){
      S.make_eigenbasis();
      PTENS_ASSRT(W.dims.size()==3);
      PTENS_ASSRT(W.dims[0]==S.obj->eblocks.size());
      PTENS_ASSRT(W.dims[1]==get_nc());
      PTENS_ASSRT(W.dims[2]==r.get_nc());
      cnine::Rtensor3_view Wt(W.mem(),W.dims[0],W.dims[2],W.dims[1],W.strides[0],W.strides[2],W.strides[1],W.dev);
      add_autobahn(get_grad(),r.get_grad(),S.obj->evecs.view2(),S.obj->eblocks,Wt);
    }

    static void add_autobahn(Ptensors1& R, const Ptensors1& x, 
      const cnine::Rtensor2_view& E, const vector<int> blocks, const cnine::Rtensor3_view& W){
      int N=R.getn();
      int K=E.n0;
      int nc=R.get_nc();
      int xnc=x.get_nc();
      int nblocks=blocks.size();

      PTENS_ASSRT(x.getn()==N);
      PTENS_ASSRT(R.tail==N*K*nc);
      PTENS_ASSRT(x.tail==N*K*xnc);
      PTENS_ASSRT(W.n0==nblocks);
      PTENS_ASSRT(W.n1==xnc);
      PTENS_ASSRT(W.n2==nc);
      PTENS_ASSRT(E.n1==K);

      auto A=cnine::Tensor<float>::zero({N,K,xnc},x.dev);
      A.view3().add_mprod(E.transp(),x.view3(K));

      auto B=cnine::Tensor<float>::zero({N,K,nc},x.dev);
      int offs=0;
      for(int b=0; b<nblocks; b++){
	for(int i=offs; i<offs+blocks[b]; i++)
	  B.view3().slice1(i).add_matmul_AA(A.view3().slice1(i),W.slice0(b));
	offs+=blocks[b];
      }

      R.view3(K).add_mprod(E,B.view3());
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
    //template<typename LAYER>
    //TransferMap overlaps(const LAYER& x){
    //return TransferMap(atoms,x.atoms);
    //}

    //SubgraphLayer0<TLAYER> transfer0(const Subgraph& _S){
    //SubgraphLayer0<TLAYER> R(G,_S,getn(),get_nc());
    //emp10(R,*this,TransferMap(atoms,R.atoms));
    //}

    //SubgraphLayer1<TLAYER> transfer1(const Subgraph& _S){
    //SubgraphLayer1<TLAYER> R(G,_S,FindPlantedSubgraphs(G,_S),get_nc());
    //emp11(R,*this,TransferMap(atoms,R.atoms));
    //}
