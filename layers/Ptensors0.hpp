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

#ifndef _ptens_Ptensors0
#define _ptens_Ptensors0

#include "Ptens_base.hpp"

//#include "Cgraph.hpp"
#include "RtensorPackB.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor0.hpp"
#include "loose_ptr.hpp"
#include "diff_class.hpp"

#include "PtensLoggedTimer.hpp"


namespace ptens{


  #ifdef _WITH_CUDA
  extern void Ptensors0_reduce0_cu(cnine::RtensorPackB& R,const cnine::RtensorPackB& x, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors0_reduce0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors0_broadcast0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const int offs, const cudaStream_t& stream);
  extern void Ptensors0_broadcast0_cu(cnine::RtensorPackB& R, const cnine::RtensorPackB& x, const AindexPack& list, const int offs, const cudaStream_t& stream);
  #endif


  class Ptensors0: public cnine::RtensorPackB, public cnine::diff_class<Ptensors0>{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    //typedef cnine::RtensorPackB RtensorPack;
    typedef cnine::RtensorPackB RtensorPackB;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    //int nc=0;
    AtomsPack atoms;
    //bool is_view=false;
    rtensor norms;


    ~Ptensors0(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      //if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0(){}

    Ptensors0(const int _nc, const int _dev=0):
      RtensorPackB(1,_nc,_dev)/*, nc(_nc)*/{}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPackB(_n, cnine::Gdims({_nc}), dummy, _dev), atoms(_n)/*, nc(_nc)*/{}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPackB(_atoms.size(), cnine::Gdims({_nc}), dummy, _dev), atoms(_atoms) /*, nc(_nc)*/{
    }


  public: // ----- Named Constructors ------------------------------------------------------------------------


    static Ptensors0 raw(const int _n, const int _nc, const int _dev=0){
      return Ptensors0(_n,_nc,cnine::fill_raw(),_dev);}

    static Ptensors0 zero(const int _n, const int _nc, const int _dev=0){
      return Ptensors0(_n,_nc,cnine::fill_zero(),_dev);}

    static Ptensors0 gaussian(const int _n, const int _nc, const int _dev=0){
      return Ptensors0(_n,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors0 gaussian(const int _n, const int _nc, const float sigma, const int _dev){
      return Ptensors0(_n,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensors0 randn(const int _n, const int _nc, const int _dev=0){
      return Ptensors0(_n,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors0 randn(const int _n, const int _nc, const float sigma, const int _dev){
      return Ptensors0(_n,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensors0 sequential(const int _n, const int _nc, const int _dev=0){
      Ptensors0 R(_n,_nc,cnine::fill_raw());
      for(int i=0; i<_n; i++) R.view1_of(i).set(i);
      return R.to_device(_dev);
    }

    static Ptensors0 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_raw(),_dev);}

    static Ptensors0 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_zero(),_dev);}

    static Ptensors0 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors0 gaussian(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensors0(_atoms,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensors0 randn(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors0 randn(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensors0(_atoms,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensors0 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors0 R(_atoms,_nc,cnine::fill_raw());
      for(int i=0; i<R.size(); i++) R.view1_of(i).set(i);
      return R.to_device(_dev);
    }

    static Ptensors0 concat(const Ptensors0& x, const Ptensors0& y){
      Ptensors0 R=Ptensors0::zero(x.atoms,x.nc+y.nc,x.dev);
      R.add_to_channels(x,0);
      R.add_to_channels(y,x.nc);
      return R;
    }


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static Ptensors0 zeros_like(const Ptensors0& x){
      return Ptensors0(RtensorPackB::zeros_like(x),x.atoms);
    }

    static Ptensors0* new_zeros_like(const Ptensors0& x){
      return new Ptensors0(RtensorPackB::zeros_like(x),x.atoms);
    }

    static Ptensors0 gaussian_like(const Ptensors0& x){
      return Ptensors0(RtensorPackB::gaussian_like(x),x.atoms);
    }

    static Ptensors0 randn_like(const Ptensors0& x){
      return Ptensors0(RtensorPackB::gaussian_like(x),x.atoms);
    }

    static Ptensors0 sequential_like(const Ptensors0& x){
      return Ptensors0(RtensorPackB::gaussian_like(x),x.atoms);
    }


    
  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x):
      RtensorPackB(x),
      cnine::diff_class<Ptensors0>(x),
      atoms(x.atoms)/*,nc(x.nc)*/{
      PTENS_COPY_WARNING();
      //#ifdef WITH_FAKE_GRAD
      //if(x.grad) grad=new Ptensors0(*grad);
      //#endif 
    }
	
    Ptensors0(Ptensors0&& x):
      RtensorPackB(std::move(x)),
      cnine::diff_class<Ptensors0>(std::move(x)),
      atoms(std::move(x.atoms))/*,nc(x.nc)*/{
      PTENS_MOVE_WARNING();
      //#ifdef WITH_FAKE_GRAD
      //grad=x.grad;
      //x.grad=nullptr;
      //#endif 
    }
    
    Ptensors0& operator=(const Ptensors0& x)=delete;


  public: // ----- Conversions -------------------------------------------------------------------------------


    //Ptensors0(cnine::RtensorPack&& x, const AtomsPack& _atoms, const int _nc):
    //RtensorPackB(std::move(x),_nc), atoms(_atoms)/*, nc(_nc)*/{}

    Ptensors0(RtensorPackB&& x, const AtomsPack& _atoms): //, const int _nc):
      RtensorPackB(std::move(x)), atoms(_atoms)/*, nc(_nc)*/{}

    Ptensors0(const rtensor& A):
      RtensorPackB(A), atoms(A.dim(0)){
      //nc=A.dim(1);
    }

    Ptensors0(const rtensor& A, const AtomsPack& _atoms):
      RtensorPackB(A), atoms(_atoms){
      //nc=A.dim(1);
    }

    #ifdef _WITH_ATEN
    Ptensors0(const at::Tensor& T):
      RtensorPackB(rtensor::regular(T)){
      assert(size()>0);
      atoms=AtomsPack(size());
      //nc=dim_of(0,0);
    }

    Ptensors0(const at::Tensor& T, const AtomsPack& _atoms):
      Ptensors0(rtensor::regular(T),_atoms){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x, const int _dev):
      RtensorPackB(x,_dev),
      atoms(x.atoms)/*,nc(x.nc)*/{}

    Ptensors0& to_device(const int _dev){
      RtensorPackB::to_device(_dev);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getn() const{
      return size();
    }

    //int get_nc() const{
    //return nc;
    //}

    AtomsPack view_of_atoms(){
      return atoms.view();
    }


    int k_of(const int i) const{
      return dim_of(i,0);
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }
    
    rtensor tensor_of(const int i) const{
      return RtensorPackB::operator()(i);
    }

    Rtensor1_view view_of(const int i) const{
      return RtensorPackB::view1_of(i);
    }

    Rtensor1_view view_of(const int i, const int offs, const int n) const{
      return RtensorPackB::view1_of(i).block(offs,n);
    }

    Rtensor1_view view_of(const int i, const vector<int>& ix) const{
      return RtensorPackB::view1_of(i);
    }

    Rtensor1_view view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      return RtensorPackB::view1_of(i).block(offs,n);
    }

    Ptensor0 operator()(const int i) const{
      return Ptensor0(tensor_of(i),atoms_of(i));
    }

    void push_back(const Ptensor0& x){
      PTENS_CPUONLY();
      if(nc==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPackB::push_back(x);
      atoms.push_back(x.atoms);
    }

    template<typename OBJ1, typename OBJ2, typename FN>
    void for_each_view(const OBJ1& x, const OBJ2& y, FN lambda){
      int N=size();
      PTENS_ASSRT(x.size()==N);
      PTENS_ASSRT(y.size()==N);
      for(int i=0; i<N; i++)
	lambda(view_of(i),x.view_of(i),y.view_of(i));
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_to_channels(const Ptensors0& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i,offs,x.nc)+=x.view_of(i);
    }

    void add_channels(const Ptensors0& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i)+=x.view_of(i,offs,nc);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPackB reduce0() const{
      TimedFn T("Ptensors0","reduce0",*this);
      RtensorPackB R(size(),Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<size(); i++)
	  R.view1_of(i).add(view_of(i));
      }

      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,0,nc,stream)));
      return R;
    }

    RtensorPackB reduce0(const int offs, const int n) const{
      TimedFn T("Ptensors0","reduce0",*this);
      RtensorPackB R(size(),Gdims(n),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<size(); i++)
	  R.view1_of(i).add(view_of(i,offs,n));
      }
      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,offs,n,stream)));
      return R;
    }


  public: // ---- Indexed reductions ---------------------------------------------------------------------------------


    RtensorPackB reduce0(const AindexPack& list) const{
      TimedFn T("Ptensors0","reduce0",*this,list,list.size()*nc);
      int N=list.size();
      cnine::array_pool<int> dims;
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view1_of(i)=view_of(list.tix(i)); // OK
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    void reduce0_back(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors0","reduce0_back",*this,x,list,list.size()*nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  view_of(list.tens(i),list.ix(i))+=x.view1_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,list,0,stream)));
    }

    // Deprecated 
    RtensorPackB reduce0(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensors0","reduce0",*this,list,list.size()*nc);
      int N=list.size();
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view1_of(i)=view_of(list.tix(i),offs,n); // OK
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,list,offs,n,stream)));
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------

    
    void broadcast0(const RtensorPackB& x){
      TimedFn T("Ptensors0","brcast0",*this,x);
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i)+=x.view1_of(i);
      }
      GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,0,stream)));
    }

    void broadcast0(const RtensorPackB& x, const int offs){
      TimedFn T("Ptensors0","brcast0",*this,x);
      if(dev==0){
	const int n=x.nc;
	for(int i=0; i<size(); i++)
	  view_of(i,offs,n).add(x.view1_of(i));
      }
      GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,offs,stream)));
    }


  public: // ---- Indexed broadcasting -------------------------------------------------------------------------------


    void broadcast0(const RtensorPackB& x, const AindexPack& list, const int offs){
      TimedFn T("Ptensors0","brcast0",*this,x,list,list.size()*nc);
      if(dev==0){
	int N=list.size();
	const int n=x.nc;
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i),offs,n)+=x.view1_of(i);
      }
      GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,list,offs,stream)));
    }

    RtensorPackB broadcast0_back(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensors0","brcast0_back",*this,list,list.size()*nc);
      int N=list.size();
      RtensorPackB R(N,Gdims(n),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view1_of(i)=view_of(list.tix(i),offs,n);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,list,offs,n,stream)));
      return R;
    }

    // deprecated 
    void broadcast0(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensors0","brcast0",*this,x,list,list.size()*nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  view_of(list.tens(i),list.ix(i))+=x.view1_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,list,0,stream)));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors0";
    }

    string repr() const{
      return "<Ptensors0[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(dev>0){
	Ptensors0 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors0& x){
      stream<<x.str(); return stream;}

  };

}


#endif 

