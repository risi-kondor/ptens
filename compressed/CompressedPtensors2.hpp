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

#ifndef _ptens_CompressedPtensors2
#define _ptens_CompressedPtensors2

#include "diff_class.hpp"
#include "Ptensors2.hpp"
#include "CompressedPtensors.hpp"
#include "CompressedGatherMatrixFactory.hpp"


namespace ptens{

  #ifdef _WITH_CUDA
  #endif 


  template<typename TYPE>
  class CompressedPtensors2: public CompressedPtensors<TYPE>, public cnine::diff_class<CompressedPtensors2<TYPE> >{
  public:

    //friend class Ptensors0<TYPE>;
    //friend class CompressedPtensors2<TYPE>;

    typedef CompressedPtensors<TYPE> BASE;
    typedef cnine::TensorView<TYPE> TENSOR;

    using cnine::diff_class<CompressedPtensors2<TYPE> >::grad;

    using BASE::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;
    using TENSOR::dev;
    using TENSOR::strides;
    using TENSOR::get_arr;
    using TENSOR::cols;
    using TENSOR::slice;

    using BASE::nc;
    using BASE::atoms;
    using BASE::size;
    //using BASE::atoms_of;
    using BASE::get_nc;


    ~CompressedPtensors2(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    CompressedPtensors2(const CompressedAtomsPack& _atoms, const TENSOR& M):
      BASE(_atoms,M){}

    CompressedPtensors2(const CompressedAtomsPack& _atoms, const int nc, const int fcode=0, const int _dev=0):
      BASE(_atoms,TENSOR({_atoms.size(),_atoms.nvecs(),_atoms.nvecs(),nc},fcode,_dev)){}


  public: // ---- Transport ----------------------------------------------------------------------------------

    
    CompressedPtensors2(const CompressedPtensors2& x, const int _dev):
      BASE(x.atoms,TENSOR(x,_dev)){}


  public: // ---- Conversions --------------------------------------------------------------------------------


    CompressedPtensors2(const CompressedAtomsPack& _atoms, const Ptensors2<TYPE>& x):
      BASE(_atoms,TENSOR({_atoms.size(),_atoms.nvecs(),_atoms.nvecs(),x.get_nc()},0,x.get_dev())){
      PTENS_ASSRT(*x.atoms.obj==*atoms->atoms);
      int N=size();
      for(int i=0; i<N; i++)
	(*this)(i).add_mprod((*atoms)(i).transp(), x(i)*((*atoms)(i))); // TODO 
    }

    Ptensors2<TYPE> uncompress(){
      Ptensors2<TYPE> R(AtomsPack(atoms->atoms),get_nc(),get_dev());
      int N=size();
      for(int i=0; i<N; i++)
	R(i).add_mprod((*atoms)(i)*(*this)(i),(*atoms)(i)); // TODO 
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    constexpr int getk(){
      return 2;
    }

    int nvecs() const{
      return dim(1);
    }

    TENSOR operator()(const int i) const{
      return slice(0,i);
    }

    TENSOR channels(const int offs, const int n) const{
      return TENSOR::slices(3,offs,n);
    }

    TENSOR as_matrix() const{
      return TENSOR::fuse(0,1).TENSOR::fuse(0,1);
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static CompressedPtensors2 linmaps(const CompressedAtomsPack& _atoms, const SOURCE& x){
      CompressedPtensors2 R(_atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const CompressedPtensors1<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0(),0);
      broadcast1(x,2*nc);
    }

    void add_linmaps(const CompressedPtensors2<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0(),0);
      broadcast1(x.reduce1(),4*nc);
      broadcast2(x,13*nc);
    }

    void add_linmaps_back(const Ptensors0<TYPE>& x){
      broadcast0_shrink(x);
    }

    void add_linmaps_back(const CompressedPtensors1<TYPE>& x){
      int nc=x.get_nc();
      broadcast0_shrink(x.reduce0(0,2*nc));
      broadcast1_shrink(x.channels(2*nc,3*nc));
    }

    void add_linmaps_back(const CompressedPtensors2<TYPE>& x){
      int nc=x.get_nc();
      broadcast0_shrink(x.reduce0_shrink(0,nc));
      broadcast1_shrink(x.reduce1_shrink(4*nc,3*nc));
      add(x.reduce2_shrink(13*nc,nc));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------

    
    template<typename SOURCE>
    static CompressedPtensors2<TYPE> gather(const CompressedAtomsPack& atoms, const SOURCE& x){
      int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
      CompressedPtensors2<TYPE> R(atoms,nc,x.get_dev());
      R.add_gather(x,LayerMap::overlaps_map(atoms.atoms(),x.atoms.atoms()));
      return R;
    }

    template<typename SOURCE>
    static CompressedPtensors2<TYPE> gather(const CompressedAtomsPack& a, const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
      CompressedPtensors1<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,map);
      return R;
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc();

      if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value){
	//auto pmap=CompressedGatherMatrixFactory<1,0,0>::gather_matrix(map,atoms,x.atoms);
	//broadcast0(x.reduce0(pmap.in()),pmap.out(),0);
      }

      if constexpr(std::is_same<SOURCE,CompressedPtensors1<TYPE> >::value){
	auto Q0=CompressedGatherMatrixFactory<2,1,0>::gather_matrix(map,atoms,x.atoms);
	auto Q1=CompressedGatherMatrixFactory<2,1,1>::gather_matrix(map,atoms,x.atoms);
	Q0.apply(channels(0,2*nc),x);
	Q1.apply(channels(2*nc,3*nc),x);
      }

      if constexpr(std::is_same<SOURCE,CompressedPtensors2<TYPE> >::value){
	auto Q0=CompressedGatherMatrixFactory<2,2,0>::gather_matrix(map,atoms,x.atoms);
	auto Q1=CompressedGatherMatrixFactory<2,2,1>::gather_matrix(map,atoms,x.atoms);
	auto Q2=CompressedGatherMatrixFactory<2,2,2>::gather_matrix(map,atoms,x.atoms);
	Q0.apply(channels(0,2*nc),x);
	Q1.apply(channels(2*nc,9*nc),x);
	Q2.apply(channels(13*nc,2*nc),x);
      }

    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const LayerMap& map){
      int nc=get_nc();

      if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value){
	//auto pmap=CompressedGatherPlanFactory::gather_matrix0(map,x.atoms,atoms,x.getk(),1);
	//broadcast0(x.reduce0(pmap.out()),pmap.in(),0);
      }

      if constexpr(std::is_same<OUTPUT,CompressedPtensors1<TYPE> >::value){
	auto Q0=CompressedGatherMatrixFactory<2,1,0>::gather_matrix(map,x.atoms,atoms);
	auto Q1=CompressedGatherMatrixFactory<2,1,1>::gather_matrix(map,x.atoms,atoms);
	Q0.apply_back(*this,x.channels(0,2*nc));
	Q1.apply_back(*this,x.channels(2*nc,3*nc));
      }

      if constexpr(std::is_same<OUTPUT,CompressedPtensors2<TYPE> >::value){
	auto Q0=CompressedGatherMatrixFactory<2,1,0>::gather_matrix(map,x.atoms,atoms);
	auto Q1=CompressedGatherMatrixFactory<2,1,1>::gather_matrix(map,x.atoms,atoms);
	auto Q2=CompressedGatherMatrixFactory<2,2,1>::gather_matrix(map,x.atoms,atoms);
	Q0.apply(*this,x.channels(0,2*nc));
	Q1.apply(*this,x.channels(2*nc,9*nc));
	Q2.apply(*this,x.channels(13*nc,2*nc));
      }

    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    TENSOR reduce0(const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc()-offs;
      TENSOR R({dim(0),2*nc},get_dev());
      R.cols(0,nc)+=channels(offs,nc).sum(1).sum(1);
      R.cols(nc,nc)+=channels(offs,nc).diag({1,2}).sum(1);
      return R;
    }

    TENSOR reduce0_shrink(const int offs, const int nc) const{
      TENSOR R({dim(0),nc},get_dev());
      R+=channels(offs,nc).sum(1).sum(1);
      R+=channels(offs+nc,nc).diag({1,2}).sum(1);
      return R;
    }

    TENSOR reduce1(const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc()-offs;
      TENSOR R({dim(0),dim(1),3*nc},get_dev());
      R.slices(2,0,nc)+=channels(offs,nc).sum(1);
      R.slices(2,nc,nc)+=channels(offs,nc).sum(2);
      R.slices(2,2*nc,nc)+=channels(offs,nc).diag({1,2});
      return R;
    }

    TENSOR reduce1_shrink(const int offs, const int nc) const{
      TENSOR R({dim(0),dim(1),nc},get_dev());
      R+=channels(offs,nc).sum(1);
      R+=channels(offs+nc,nc).sum(2);
      R+=channels(offs+2*nc,nc).diag({1,2});
      return R;
    }

    TENSOR reduce2(const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc()-offs;
      return channels(offs,nc);
    }

    TENSOR reduce2_shrink(const int offs, const int nc) const{
      TENSOR R({dim(0),dim(1),dim(1),nc},get_dev());
      R+=channels(offs,nc);
      R+=channels(offs+nc,nc).transp(1,2);
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      int nc=X.dim(1);
      channels(offs,nc)+=X.insert_dim(1,nvecs()).insert_dim(1,nvecs());
      channels(offs+nc,nc).diag({1,2})+=X.insert_dim(1,nvecs());
    }

    void broadcast0_shrink(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      int nc=X.dim(1)/2; // different from Ptensors2
      channels(offs,nc)+=X.slices(1,0,nc).insert_dim(1,nvecs()).insert_dim(1,nvecs());
      channels(offs,nc).diag({1,2})+=X.slices(1,0,nc).insert_dim(1,nvecs());
    }

    void broadcast1(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      int nc=X.dim(1);
      channels(offs,nc)+=X.insert_dim(1,nvecs());
      channels(offs+nc,nc)+=X.insert_dim(2,nvecs());
      channels(offs+2*nc,nc).diag({1,2})+=X;
    }

    void broadcast1_shrink(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      int nc=X.dim(1)/3;
      channels(offs,nc)+=X.slices(2,0,nc).insert_dim(1,nvecs());
      channels(offs,nc)+=X.slices(2,0,nc).insert_dim(2,nvecs());
      channels(offs,nc).diag({1,2})+=X.slices(2,0,nc);
    }

    void broadcast2(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      int nc=X.dim(1);
      channels(offs,nc)+=X;
      channels(offs+nc,nc)+=X.transp(1,2);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "CompressedPtensors2";
    }

    string repr() const{
      return "<CompressedPtensors2[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	CompressedPtensors2 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<(*this)(i).str(indent);
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CompressedPtensors2& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
