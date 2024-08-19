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

#ifndef _ptens_CompressedPtensors1
#define _ptens_CompressedPtensors1

#include "diff_class.hpp"
#include "Ptensors1.hpp"
#include "CompressedPtensors.hpp"
//#include "BlockCsparseMatrix.hpp"
#include "CompressedGatherMatrixFactory.hpp"


namespace ptens{

  template<typename TYPE>
  class CompressedPtensors2;


  template<typename TYPE>
  class CompressedPtensors1: public CompressedPtensors<TYPE>, public cnine::diff_class<CompressedPtensors1<TYPE> >{
  public:

    //friend class Ptensors0<TYPE>;
    //friend class CompressedPtensors2<TYPE>;

    typedef CompressedPtensors<TYPE> BASE;
    typedef cnine::TensorView<TYPE> TENSOR;

    using cnine::diff_class<CompressedPtensors1<TYPE> >::grad;

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


    ~CompressedPtensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    CompressedPtensors1(const CompressedAtomsPack& _atoms, const TENSOR& M):
      BASE(_atoms,M){}

    CompressedPtensors1(const CompressedAtomsPack& _atoms, const int nc, const int fcode=0, const int _dev=0):
      BASE(_atoms,TENSOR({_atoms.size(),_atoms.nvecs(),nc},fcode,_dev)){}


  public: // ---- Transport ----------------------------------------------------------------------------------

    
    CompressedPtensors1(const CompressedPtensors1& x, const int _dev):
      BASE(x.atoms,TENSOR(x,_dev)){}


  public: // ---- Conversions --------------------------------------------------------------------------------


    CompressedPtensors1(const CompressedAtomsPack& _atoms, const Ptensors1<TYPE>& x):
      BASE(_atoms,TENSOR({_atoms.size(),_atoms.nvecs(),x.get_nc()},0,x.get_dev())){
      PTENS_ASSRT(x.atoms==atoms.atoms());
      int N=size();
      for(int i=0; i<N; i++)
	(*this)(i).add_mprod(atoms.basis(i).transp(), x(i));
    }

    Ptensors1<TYPE> uncompress(){
      Ptensors1<TYPE> R(AtomsPack(atoms->atoms),get_nc(),get_dev());
      int N=size();
      for(int i=0; i<N; i++)
	R(i).add_mprod(atoms.basis(i),(*this)(i));
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    constexpr int getk() const{
      return 1;
    }

    int nvecs() const{
      return dim(1);
    }

    TENSOR operator()(const int i) const{
      return slice(0,i);
    }

    TENSOR channels(const int offs, const int n) const{
      return TENSOR::slices(2,offs,n);
    }

    TENSOR as_matrix() const{
      return TENSOR::fuse(0,1);
    }

    TENSOR as_matrix(const int nc) const{
      return TENSOR::fuse(0,1);
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static CompressedPtensors1 linmaps(const CompressedAtomsPack& _atoms, const SOURCE& x){
      CompressedPtensors1 R(_atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const CompressedPtensors1<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      channels(nc,nc)+=x;
    }

    void add_linmaps(const CompressedPtensors2<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      channels(2*nc,3*nc)+=x.reduce1();
    }

    void add_linmaps_back(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps_back(const CompressedPtensors1<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0(0,nc));
      add(x.channels(nc,nc));
    }

    void add_linmaps_back(const CompressedPtensors2<TYPE>& x){
      int nc=x.get_nc();
      add(x.reduce0_shrink(0,nc));
      add(x.reduce1_shrink(2*nc,nc));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------

    
    template<typename SOURCE>
    static CompressedPtensors1<TYPE> gather(const CompressedAtomsPack& atoms, const SOURCE& x){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      CompressedPtensors1<TYPE> R(atoms,nc,x.get_dev());
      R.add_gather(x,LayerMap::overlaps_map(atoms.atoms(),x.atoms.atoms()));
      return R;
    }

    template<typename SOURCE>
    static CompressedPtensors1<TYPE> gather(const CompressedAtomsPack& a, const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
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
	auto Q0=CompressedGatherMatrixFactory<1,1,0>::gather_matrix(map,atoms,x.atoms);
	auto Q1=CompressedGatherMatrixFactory<1,1,1>::gather_matrix(map,atoms,x.atoms);
	Q0.apply(channels(0,nc),x);
	Q1.apply(channels(nc,nc),x);
      }

      if constexpr(std::is_same<SOURCE,CompressedPtensors2<TYPE> >::value){
	auto Q0=CompressedGatherMatrixFactory<1,2,0>::gather_matrix(map,atoms,x.atoms);
	auto Q1=CompressedGatherMatrixFactory<1,2,1>::gather_matrix(map,atoms,x.atoms);
	Q0.apply(channels(0,2*nc),x);
	Q1.apply(channels(2*nc,3*nc),x);
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
	auto Q0=CompressedGatherMatrixFactory<1,1,0>::gather_matrix(map,x.atoms,atoms);
	auto Q1=CompressedGatherMatrixFactory<1,1,1>::gather_matrix(map,x.atoms,atoms);
	Q0.apply_back(*this,x.channels(0,nc));
	Q1.apply_back(*this,x.channels(nc,nc));
      }

      if constexpr(std::is_same<OUTPUT,CompressedPtensors2<TYPE> >::value){
	auto Q0=CompressedGatherMatrixFactory<2,1,0>::gather_matrix(map,x.atoms,atoms);
	auto Q1=CompressedGatherMatrixFactory<2,1,1>::gather_matrix(map,x.atoms,atoms);
	Q0.apply(*this,x.channels(0,2*nc));
	Q1.apply(*this,x.channels(2*nc,3*nc));
      }

    }


  public: // ---- Reductions ---------------------------------------------------------------------------------

    
    TENSOR reduce0(const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc()-offs;
      return channels(offs,nc).sum(1);
    }

    TENSOR reduce1(const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc()-offs;
      return channels(offs,nc);
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      channels(offs,X.dim(1))+=X.insert_dim(1,nvecs());
    }

    void broadcast1(const TENSOR& X, const int offs=0){
      PTENS_ASSRT(X.dim(0)==dim(0));
      PTENS_ASSRT(X.dim(1)==dim(1));
      channels(offs,X.dim(2))+=X;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "CompressedPtensors1";
    }

    string repr() const{
      return "<CompressedPtensors1[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	CompressedPtensors1 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"CPtensor1"<<atoms.atoms(i)<<":"<<endl;
	oss<<(*this)(i).to_string(indent+"  ");
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CompressedPtensors1& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
