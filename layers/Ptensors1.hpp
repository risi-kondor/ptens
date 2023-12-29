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

#ifndef _ptens_Ptensors1
#define _ptens_Ptensors1

#include "diff_class.hpp"
#include "Rtensor2_view.hpp"
#include "Ptensors.hpp"
#include "Ptensor1.hpp"
#include "PtensLoggedTimer.hpp"


namespace ptens{


  #ifdef _WITH_CUDA
  extern void Ptensors1_reduce0_cu(cnine::RtensorPackB& R,const cnine::RtensorPackB& x, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_reduce0n_cu(cnine::RtensorPackB& R,const cnine::RtensorPackB& x, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_reduce0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_reduce0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_reduce1_cu(cnine::RtensorPackB& R,const cnine::RtensorPackB& x, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_reduce1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_broadcast0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const int offs, const cudaStream_t& stream);
  extern void Ptensors1_broadcast0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const int offs, const cudaStream_t& stream);
  extern void Ptensors1_broadcast0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, const int offs, const cudaStream_t& stream);
  extern void Ptensors1_broadcast0n_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, const int offs, const cudaStream_t& stream);
  extern void Ptensors1_broadcast1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const int offs, const cudaStream_t& stream);
  extern void Ptensors1_broadcast1_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, const int offs, const cudaStream_t& stream);
  #endif


  class Ptensors1: public Ptensors, public cnine::diff_class<Ptensors1>{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::IntTensor itensor;
    typedef cnine::IntTensor IntTensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;
    typedef cnine::RtensorPackB RtensorPackB;
    typedef cnine::Tensor<int> ITENSOR;

    using Ptensors::Ptensors;

    //AtomsPack atoms;
    //rtensor norms;


    ~Ptensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1(){}

    Ptensors1(const int _nc, const int _dev=0):
      Ptensors(2,_nc,_dev){}
    //RtensorPackB(2,_nc,_dev){}

    // this should be improved
    Ptensors1(const AtomsPack& _atoms, const int _nc, const int _dev=0):
      Ptensors1(_nc,_dev){
      atoms=_atoms;
    }

    Ptensors1(const AtomsPack& _atoms, const int _nc, const cnine::fill_zero& dummy, const int _dev=0):
      Ptensors1(zero(_atoms,_nc,_dev)){}

    Ptensors1(const cnine::Tensor<int>& M, const int _nc, const cnine::fill_zero& dummy, const int _dev=0):
      Ptensors(AtomsPack(M), cnine::Gdims({M.dims[1],_nc}), dummy, _dev){}


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    [[deprecated]]
    Ptensors1(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Ptensors(_atoms, cnine::Gdims({_nc}), dummy, _dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors1(const int _n, const int _k, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Ptensors(AtomsPack(_n,_k),{_k,_nc},dummy,_dev){}


  public: // ----- Constructors ------------------------------------------------------------------------------


    static Ptensors1 raw(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors1(_n,_k,_nc,cnine::fill_raw(),_dev);}

    static Ptensors1 zero(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors1(_n,_k,_nc,cnine::fill_zero(),_dev);}

    static Ptensors1 gaussian(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors1(_n,_k,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors1 gaussian(const int _n, const int _k, const int _nc, const float sigma, const int _dev){
      return Ptensors1(_n,_k,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensors1 randn(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors1(_n,_k,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors1 randn(const int _n, const int _k, const int _nc, const float sigma, const int _dev){
      return Ptensors1(_n,_k,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensors1 sequential(const int _n, const int _k, const int _nc, const int _dev=0){
      Ptensors1 R(_n,_k,_nc,cnine::fill_raw(),0);
      for(int i=0; i<_n; i++)
	R.view2_of(i).set(i); 
      R.to_device(_dev);
      return R;
    }


    static Ptensors1 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_atoms,_nc,_dev);
      R.reserve(_atoms.tsize1()*_nc);
      R.dir=IntTensor::raw({_atoms.size(),3});
      R.tail=0;
      for(int i=0; i<_atoms.size(); i++){
	R.dir.set_row(i,{R.tail,_atoms.size_of(i),_nc});
	R.tail+=_atoms.size_of(i)*_nc;
      }
      return R;
    }

    static Ptensors1 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_atoms,_nc,_dev);
      R.reserve_zero(_atoms.tsize1()*_nc);
      R.dir=IntTensor::raw({_atoms.size(),3});
      R.tail=0;
      for(int i=0; i<_atoms.size(); i++){
	R.dir.set_row(i,{R.tail,_atoms.size_of(i),_nc});
	R.tail+=_atoms.size_of(i)*_nc;
      }
      return R;
    }

    static Ptensors1 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_nc,0);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor1::gaussian(_atoms(i),_nc));
      R.to_device(_dev);
      return R;
    }

    static Ptensors1 gaussian(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      Ptensors1 R(_nc,0);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor1::gaussian(_atoms(i),_nc,sigma,0));
      R.to_device(_dev);
      return R;
    }

    static Ptensors1 randn(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors1::gaussian(_atoms,_nc,_dev);
    }

    static Ptensors1 randn(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensors1::gaussian(_atoms,_nc,sigma,_dev);
    }

    static Ptensors1 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_nc,0);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor1::sequential(_atoms(i),_nc));
      R.to_device(_dev);
      return R;
    }


    static Ptensors1 concat(const Ptensors1& x, const Ptensors1& y){
      Ptensors1 R=Ptensors1::zero(x.atoms,x.nc+y.nc,x.dev);
      R.add_to_channels(x,0);
      R.add_to_channels(y,x.nc);
      return R;
    }


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors1(const Ptensors1& x):
      Ptensors(x),
      cnine::diff_class<Ptensors1>(x){
      PTENS_COPY_WARNING();
    }
	
    Ptensors1(Ptensors1&& x):
      Ptensors(std::move(x)),
      cnine::diff_class<Ptensors1>(std::move(x)){
      PTENS_MOVE_WARNING();
    }

    Ptensors1& operator=(const Ptensors1& x)=delete;


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static Ptensors1 zeros_like(const Ptensors1& x){
      return Ptensors1(RtensorPackB::zeros_like(x),x.atoms);
    }

    static Ptensors1 zeros_like(const Ptensors1& x, const int _nc){
      return Ptensors1(RtensorPackB::zeros_like(x,_nc),x.atoms);
    }

    static Ptensors1* new_zeros_like(const Ptensors1& x){
      return new Ptensors1(RtensorPackB::zeros_like(x),x.atoms);
    }

   static Ptensors1 gaussian_like(const Ptensors1& x){
     return Ptensors1(RtensorPackB::gaussian_like(x),x.atoms);
    }

    static Ptensors1 randn_like(const Ptensors1& x){
      return Ptensors1(RtensorPackB::gaussian_like(x),x.atoms);
    }

    static Ptensors1 sequential_like(const Ptensors1& x){
      return Ptensors1(RtensorPackB::sequential_like(x),x.atoms);
    }

    static Ptensors1 like(const Ptensors1& x, const cnine::RtensorA& M){
      return Ptensors1(M,x.atoms);
    }
      

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors1(const rtensor& A, const AtomsPack& _atoms):
      Ptensors(RtensorPackB(A,_atoms.dims1(A.dim(1))),_atoms){}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors1& to_device(const int _dev){
      Ptensors::to_device(_dev);
      return *this;
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    //AtomsPack& get_atomsref(){
    //return atoms;
    //}

    int k_of(const int i) const{
      return dim_of(i,0);
    }

    Rtensor2_view view_of(const int i) const{
      return RtensorPackB::view2_of(i);
    }

    Rtensor2_view view_of(const int i, const int offs, const int n) const{
      return RtensorPackB::view2_of(i).block(0,offs,-1,n);
    }

    Rtensor2_view constk_view_of(const int i) const{
      return Rtensor2_view(arr+nc*constk*i,constk,nc,nc,1,dev);
    }

    Rtensor2_view constk_view_of(const int i, const int offs, const int n) const{
      return Rtensor2_view(arr+nc*constk*i+offs,constk,n,nc,1,dev);
    }

    Ptensor1_xview view_of(const int i, const vector<int>& ix) const{
      vector<int> v=headers(i);
      PTENS_ASSRT(v.size()==3);
      if(dev==1) return Ptensor1_xview(arrg+v[0],v[2],v[2],1,ix,1);
      return Ptensor1_xview(arr+v[0],v[2],v[2],1,ix,0);
    }

    Ptensor1_xview view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      vector<int> v=headers(i);
      PTENS_ASSRT(v.size()==3);
      if(dev==1) return Ptensor1_xview(arrg+v[0]+offs,n,v[2],1,ix,1);
      return Ptensor1_xview(arr+v[0]+offs,n,v[2],1,ix,0);
    }

    Ptensor1 operator()(const int i) const{
      return Ptensor1(tensor_of(i),atoms_of(i));
    }

    // only works if each ref domain is of size K
    cnine::Rtensor3_view view3(const int K) const{
      int n=getn();
      int nc=get_nc();
      PTENS_ASSRT(tail==n*K*nc);
      return cnine::Rtensor3_view(get_arr(),n,K,nc,K*nc,nc,1);
    }

    int push_back(const Ptensor1& x){
      if(size()==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPackB::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }

    template<typename OBJ1, typename OBJ2, typename FN>
    void for_each_view(const OBJ1& x, const OBJ2& y, FN lambda){
      int N=size();
      PTENS_ASSRT(x.size()==N);
      PTENS_ASSRT(y.size()==N);
      for(int i=0; i<N; i++)
	lambda(view_of(i),x.view_of(i),y.view_of(i));
    }

    Ptensors1 permute(const cnine::permutation& pi){
      return Ptensors1(*this,atoms.permute(pi));
    }


  public: // ---- Concatenation ------------------------------------------------------------------------------


    static Ptensors1 cat(const vector<reference_wrapper<Ptensors1> >& list){
      vector<reference_wrapper<AtomsPack> > v;
      for(auto& p:list)
	v.push_back(p.get().atoms);
      return Ptensors1(cnine::RtensorPackB::cat
	(cnine::mapcar<reference_wrapper<Ptensors1>,reference_wrapper<RtensorPackB> >
	  (list,[](const reference_wrapper<Ptensors1>& x){
	    return reference_wrapper<RtensorPackB>(x.get());})),AtomsPack::cat(v));
    }

    static Ptensors1 sum(const vector<reference_wrapper<Ptensors1> >& list){
      if(list.size()==0) return Ptensors1();
      Ptensors1 R(list[0].get());
      for(int i=1; i<list.size(); i++)
	R.add(list[i].get());
      return R;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_to_channels(const Ptensors1& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i,offs,x.nc)+=x.view_of(i);
    }

    void add_channels(const Ptensors1& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i)+=x.view_of(i,offs,nc);
    }

    /*
    void add_mprod(const Ptensors1& x, const rtensor& y){
      PTENS_CPUONLY();
      PTENS_ASSRT(x.size()==size());
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i).add_matmul_AA(x.view_of(i),y.view2());
      }else{
	view_as_matrix().add_mprod(x.view_as_matrix(),y);
      }
    }

    void add_mprod_back0(const Ptensors1& g, const rtensor& y){
      PTENS_CPUONLY();
      PTENS_ASSRT(g.size()==size());
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i).add_matmul_AT(g.view_of(i),y.view2());
      }else{
	view_as_matrix().add_Mprod_AT(g.view_as_matrix(),y);
      }
    }

    void add_mprod_back1_to(rtensor& r, const Ptensors1& x) const{
      PTENS_CPUONLY();
      PTENS_ASSRT(x.size()==size());
      if(dev==0){
	for(int i=0; i<size(); i++)
	  r.view2().add_matmul_TA(x.view_of(i),view_of(i));
      }else{
	r.add_Mprod_TA(x.view_as_matrix(),view_as_matrix());
      }
    }
    */

    Ptensors1 scale_channels(const rtensor& y) const{
      return Ptensors1(RtensorPackB::scale_channels(y.view1()),atoms);
    }

 
  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPackB reduce0() const{
      TimedFn T("Ptensors1","reduce0",*this);
      RtensorPackB R(size(),Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i).sum0_into(R.view1_of(i));
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,0,nc,stream)));
      return R;
    }

    RtensorPackB reduce0_n() const{
      TimedFn T("Ptensors1","reduce0_n",*this);
      RtensorPackB R(size(),Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i).avg0_into(R.view1_of(i));
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0n_cu(R,*this,0,nc,stream)));
      return R;
    }

    RtensorPackB reduce0(const int offs, const int n) const{
      TimedFn T("Ptensors1","reduce0",*this);
      RtensorPackB R(size(),Gdims(n),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i,offs,n).sum0_into(R.view1_of(i));
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,offs,n,stream)));
      return R;
    }

    RtensorPackB reduce1() const{
      TimedFn T("Ptensors1","reduce1",*this);
      return *this;
    }

    RtensorPackB reduce1(const int offs, const int n) const{
      TimedFn T("Ptensors1","reduce1",*this);
      cnine::array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),n}));
      RtensorPackB R(dims,cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<size(); i++){
	  R.view2_of(i)+=view_of(i,offs,n);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,offs,n,stream)));
      return R;
    }


  public: // ---- Indexed Reductions --------------------------------------------------------------------------------------


    RtensorPackB reduce0(const AindexPack& list) const{
      TimedFn T("Ptensors1","reduce0",*this,list,list.count1*nc);
      int N=list.size();
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).sum0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    RtensorPackB reduce0(const cnine::TensorView<int>& indices) const{
      int N=indices.dim(0);
      int k=indices.dim(1)-1;
      TimedFn T("Ptensors1","reduce0G",*this,indices,N*k*nc);
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  auto src=view_of(indices(i,0));
	  for(int j=0; j<k; j++){
	    R.view1_of(i)+=src.slice0(indices(i,j+1));
	  }
	}
      }
      CNINE_CPUONLY();
      //GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    void reduce0_back(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors1","reduce0_back",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i))+=repeat0(x.view1_of(i),list.nix(i));
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,list,0,stream)));
    }


    RtensorPackB reduce1(const AindexPack& list) const{
      TimedFn T("Ptensors1","reduce1",*this,list,list.count1*nc);
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),nc});
      RtensorPackB R(dims,cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view2_of(i)+=view_of(list.tens(i),list.ix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    RtensorPackB reduce1(const cnine::TensorView<int>& indices) const{
      int N=indices.dim(0);
      int k=indices.dim(1)-1;
      TimedFn T("Ptensors1","reduce1",*this,indices,N*k*nc);
      RtensorPackB R(N,cnine::Gdims(k,nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  auto src=view_of(indices(i,0));
	  auto dest=R.view2_of(i);
	  for(int j=0; j<k; j++)
	    dest.slice0(j)+=src.slice0(indices(i,j+1));
	}
      }
      CNINE_CPUONLY();
      //GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    void reduce1_back(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors1","reduce1_back",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.dim_of(i,0)==0) continue;
	  view_of(list.tens(i),list.ix(i))+=x.view2_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,list,0,stream)));
    }




    RtensorPackB reduce0_n(const AindexPack& list) const{
      TimedFn T("Ptensors1","reduce0_n",*this,list,list.count1*nc);
      int N=list.size();
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).avg0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0n_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    // deprecated 
    //[[deprecated]]
    RtensorPackB reduce0(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensors1","reduce0",*this,list,list.count1*n);
      int N=list.size();
      RtensorPackB R(N,Gdims(n),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n).sum0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,list,offs,n,stream)));
      return R;
    }

    // deprecated 
    //[[deprecated]]
    RtensorPackB reduce1(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensors1","reduce1",*this,list,list.count1*n);
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),n});
      RtensorPackB R(dims,cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view2_of(i)+=view_of(list.tens(i),list.ix(i),offs,n);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,list,offs,n,stream)));
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const RtensorPackB& x){
      TimedFn T("Ptensors1","brcast0",*this,x);
      if(dev==0){
	for(int i=0; i<size(); i++){
	  view_of(i)+=repeat0(x.view1_of(i),k_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,0,stream)));
    }

    void broadcast0_n(const RtensorPackB& x){
      TimedFn T("Ptensors1","brcast0",*this,x);
      if(dev==0){
	for(int i=0; i<size(); i++){
	  view_of(i).add(repeat0(x.view1_of(i),k_of(i)),1.0/((float)k_of(i)));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0n_cu(*this,x,0,stream)));
    }

    void broadcast0(const RtensorPackB& x, const int offs){
      TimedFn T("Ptensors1","brcast0",*this,x);
      const int n=x.nc;
      if(dev==0){
	for(int i=0; i<size(); i++){
	  view_of(i,offs,n)+=repeat0(x.view1_of(i),k_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,offs,stream)));
    }

    void broadcast1(const RtensorPackB& x){
      TimedFn T("Ptensors1","brcast1",*this,x);
      if(dev==0){
	for(int i=0; i<size(); i++){
	  view_of(i)+=x.view2_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,0,stream)));
    }

    void broadcast1(const RtensorPackB& x, const int offs){
      TimedFn T("Ptensors1","brcast1",*this,x);
      if(dev==0){
	const int n=x.nc;
	for(int i=0; i<size(); i++){
	  view_of(i,offs,n)+=x.view2_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,offs,stream)));
    }


  public: // ---- Indexed broadcasting ------------------------------------------------------------------------------


    void broadcast0(const RtensorPackB& x, const AindexPack& list, const int offs){
      TimedFn T("Ptensors1","brcast0",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	const int n=x.nc;
	for(int i=0; i<N; i++){
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view1_of(i),list.nix(i));
	  //cout<<"qqq"<<view_of(list.tens(i))<<endl;
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,list,offs,stream)));
    }

    void broadcast0(const RtensorPackB& x, const cnine::TensorView<int>& indices, 
      const cnine::GatherMap& bmap, const int offs){
      int N=indices.dim(0);
      int k=indices.dim(1)-1;
      TimedFn T("Ptensors1","brcast0G",*this,x,indices,N*k*x.nc);
      if(dev==0){
	const int n=x.nc;
	for(int i=0; i<N; i++){
	  auto dest=view_of(indices(i,0),offs,n);
	  for(int j=0; j<k; j++)
	    dest.slice0(indices(i,j+1))+=x.view1_of(i);
	}
      }
      CNINE_CPUONLY();
      //GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,list,offs,stream)));
    }

    RtensorPackB broadcast0_back(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensors1","brcast0_back",*this,list,list.count1*n);
      int N=list.size();
      RtensorPackB R(N,Gdims(n),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n).sum0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,list,offs,n,stream)));
      return R;
    }


    void broadcast1(const RtensorPackB& x, const AindexPack& list, const int offs){
      TimedFn T("Ptensors1","brcast1",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	const int n=x.nc;
	for(int i=0; i<N; i++){
	  if(x.dim_of(i,0)==0) continue;
	  //cout<<"<-"<<i<<" "<<list.tens(i)<<" "<<offs<<" "<<n<<" :";
	  //auto v=list.ix(i);
	  //for(auto p:v) cout<<p;
	  //cout<<endl;
	  //if(list.tens(i)==1) cout<<"***"<<view_of(1)<<endl;
	  view_of(list.tens(i),list.ix(i),offs,n)+=x.view2_of(i);
	  //if(list.tens(i)==1) cout<<view_of(1)<<endl;
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,list,offs,stream)));
    }

    void broadcast1(const RtensorPackB& x, const cnine::TensorView<int>& indices, 
      const cnine::GatherMap& bmap, const int offs){
      int N=indices.dim(0);
      int k=indices.dim(1)-1;
      TimedFn T("Ptensors1","brcast1G",*this,x,indices,N*k*x.nc);
      if(dev==0){
	const int n=x.nc;
	for(int i=0; i<N; i++){
	  auto src=x.view2_of(i);
	  auto dest=view_of(indices(i,0),offs,n);
	  for(int j=0; j<k; j++)
	    dest.slice0(indices(i,j+1))+=src.slice0(j);
	}
      }
      CNINE_CPUONLY();
      //GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,list,offs,stream)));
    }

    RtensorPackB broadcast1_back(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensors1","brcast1_back",*this,list,list.count1*n);
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),n});
      RtensorPackB R(dims,cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view2_of(i)+=view_of(list.tens(i),list.ix(i),offs,n);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,list,offs,n,stream)));
      return R;
    }



    void broadcast0_n(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors1","brcast0_n",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i)).add(repeat0(x.view1_of(i),list.nix(i)),1.0/((float)list.nix(i))); // check this
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0n_cu(*this,x,list,0,stream)));
    }

    // deprecated 
    //[[deprecated]]
    void broadcast0(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors1","brcast0",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i))+=repeat0(x.view1_of(i),list.nix(i));
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,list,0,stream)));
    }

    // deprecated 
    //[[deprecated]]
    void broadcast1(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors1","brcast1",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.dim_of(i,0)==0) continue;
	  view_of(list.tens(i),list.ix(i))+=x.view2_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,list,0,stream)));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors1";
    }

    string repr() const{
      if(dev==0) return "<Ptensors1[N="+to_string(size())+"]>";
      else return "<Ptensors1[N="+to_string(size())+"][G]>";
    }

    string str(const string indent="") const{
      if(dev>0){
	Ptensors1 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      //oss<<atoms<<endl;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
	//oss<<indent<<"Ptensor "<<i<<" "<<Atoms(atoms(i))<<":"<<endl;
	//oss<<RtensorPackB::operator()(i).str()<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors1& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
