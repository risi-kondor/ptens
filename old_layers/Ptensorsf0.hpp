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

#ifndef _ptens_Ptensorsf0
#define _ptens_Ptensorsf0

#include "Ptens_base.hpp"

#include "Tensor.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "loose_ptr.hpp"
#include "diff_class.hpp"

#include "PtensLoggedTimer.hpp"


namespace ptens{


  class Ptensorsf0: public cnine::Tensor<float>, public cnine::diff_class<Ptensorsf0>{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::IntTensor itensor;
    typedef cnine::Tensor<float> Tensor;
    //typedef cnine::RtensorPackB RtensorPack;
    //typedef cnine::RtensorPackB RtensorPackB;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    //int nc=0;
    //AtomsPack atoms;
    Tensor<int> atoms
    //bool is_view=false;


    ~Ptensorsf0(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      //if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //Ptensorsf0(const int _nc, const int _dev=0):
    //Tensor(1,_nc,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensorsf0(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Tensor(Gdims({_n,_nc}), dummy, _dev), atoms({_n},cnine::fill_sequential()){}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ptensorsf0(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
    //RtensorPackB(_atoms.size(), cnine::Gdims({_nc}), dummy, _dev), atoms(_atoms) /*, nc(_nc)*/{
    //}


  public: // ----- Named Constructors ------------------------------------------------------------------------


    static Ptensorsf0 raw(const int _n, const int _nc, const int _dev=0){
      return Ptensorsf0(_n,_nc,cnine::fill_raw(),_dev);}

    static Ptensorsf0 zero(const int _n, const int _nc, const int _dev=0){
      return Ptensorsf0(_n,_nc,cnine::fill_zero(),_dev);}

    static Ptensorsf0 gaussian(const int _n, const int _nc, const int _dev=0){
      return Ptensorsf0(_n,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensorsf0 gaussian(const int _n, const int _nc, const float sigma, const int _dev){
      return Ptensorsf0(_n,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensorsf0 randn(const int _n, const int _nc, const int _dev=0){
      return Ptensorsf0(_n,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensorsf0 randn(const int _n, const int _nc, const float sigma, const int _dev){
      return Ptensorsf0(_n,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensorsf0 sequential(const int _n, const int _nc, const int _dev=0){
      Ptensorsf0 R(_n,_nc,cnine::fill_raw());
      for(int i=0; i<_n; i++) R.view1_of(i).set(i);
      return R.to_device(_dev);
    }

    /*
    static Ptensorsf0 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensorsf0(_atoms,_nc,cnine::fill_raw(),_dev);}

    static Ptensorsf0 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensorsf0(_atoms,_nc,cnine::fill_zero(),_dev);}

    static Ptensorsf0 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensorsf0(_atoms,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensorsf0 gaussian(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensorsf0(_atoms,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensorsf0 randn(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensorsf0(_atoms,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensorsf0 randn(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensorsf0(_atoms,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensorsf0 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensorsf0 R(_atoms,_nc,cnine::fill_raw());
      for(int i=0; i<R.size(); i++) R.view1_of(i).set(i);
      return R.to_device(_dev);
    }

    static Ptensorsf0 concat(const Ptensorsf0& x, const Ptensorsf0& y){
      Ptensorsf0 R=Ptensorsf0::zero(x.atoms,x.nc+y.nc,x.dev);
      R.add_to_channels(x,0);
      R.add_to_channels(y,x.nc);
      return R;
    }
    */


  public: // ---- Spawning -----------------------------------------------------------------------------------

    /*
    static Ptensorsf0 zeros_like(const Ptensorsf0& x){
      return Ptensorsf0(RtensorPackB::zeros_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf0* new_zeros_like(const Ptensorsf0& x){
      return new Ptensorsf0(RtensorPackB::zeros_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf0 gaussian_like(const Ptensorsf0& x){
      return Ptensorsf0(RtensorPackB::gaussian_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf0 randn_like(const Ptensorsf0& x){
      return Ptensorsf0(RtensorPackB::gaussian_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf0 sequential_like(const Ptensorsf0& x){
      return Ptensorsf0(RtensorPackB::gaussian_like(x),x.atoms);//,x.nc);
    }
    */

    
  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensorsf0(const Ptensorsf0& x):
      Tensor(x),
      cnine::diff_class<Ptensorsf0>(x),
      atoms(x.atoms){
      PTENS_COPY_WARNING();
    }
	
    Ptensorsf0(Ptensorsf0&& x):
      RtensorPackB(std::move(x)),
      cnine::diff_class<Ptensorsf0>(std::move(x)),
      atoms(std::move(x.atoms)){
      PTENS_MOVE_WARNING();
    }
    
    Ptensorsf0& operator=(const Ptensorsf0& x)=delete;


  public: // ----- Conversions -------------------------------------------------------------------------------


    //Ptensorsf0(cnine::RtensorPack&& x, const AtomsPack& _atoms, const int _nc):
    //RtensorPackB(std::move(x),_nc), atoms(_atoms)/*, nc(_nc)*/{}

    //Ptensorsf0(RtensorPackB&& x, const AtomsPack& _atoms): //, const int _nc):
    //RtensorPackB(std::move(x)), atoms(_atoms)/*, nc(_nc)*/{}

    //Ptensorsf0(const rtensor& A):
    //RtensorPackB(A), atoms(A.dim(0)){
    //nc=A.dim(1);
    //}

    //Ptensorsf0(const rtensor& A, const AtomsPack& _atoms):
    //RtensorPackB(A), atoms(_atoms){
    //nc=A.dim(1);
    //}

    #ifdef _WITH_ATEN
    //Ptensorsf0(const at::Tensor& T):
    //RtensorPackB(rtensor::regular(T)){
    //assert(size()>0);
    //atoms=AtomsPack(size());
    //nc=dim_of(0,0);
    //}

    //Ptensorsf0(const at::Tensor& T, const AtomsPack& _atoms):
    //Ptensorsf0(rtensor::regular(T),_atoms){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensorsf0(const Ptensorsf0& x, const int _dev):
      RtensorPackB(x,_dev),
      atoms(x.atoms){}

    Ptensorsf0& to_device(const int _dev){
      RtensorPackB::to_device(_dev);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //int get_nc() const{
    //return nc;
    //}

    //AtomsPack view_of_atoms(){
    //return atoms.view();
    //}

    int size() const{
      return dims(0);
    }

    int getn() const{
      return dims(0);
    }

    int get_nc() const{
      return dims(1);
    }

    //int k_of(const int i) const{
    //return dim_of(i,0);
    //}

    Atoms atoms_of(const int i) const{
      return atoms(i);
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


    void add_to_channels(const Ptensorsf0& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i,offs,x.nc)+=x.view_of(i);
    }

    void add_channels(const Ptensorsf0& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i)+=x.view_of(i,offs,nc);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    Tensor reduce0() const{
      TimedFn T("Ptensorsf0","reduce0",*this);
      return *this;
    }

    RtensorPackB reduce0(const int offs, const int n) const{
      TimedFn T("Ptensorsf0","reduce0",*this);
      return this->cols(offs,offs+n);
    }

    /*
    RtensorPackB reduce0(const AindexPack& list) const{
      TimedFn T("Ptensorsf0","reduce0",*this,list,list.size()*nc);
      int N=list.size();
      cnine::array_pool<int> dims;
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view1_of(i)=view_of(list.tix(i)); // OK
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf0_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    RtensorPackB reduce0(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensorsf0","reduce0",*this,list,list.size()*nc);
      int N=list.size();
      RtensorPackB R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view1_of(i)=view_of(list.tix(i),offs,n); // OK
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf0_reduce0_cu(R,*this,list,offs,n,stream)));
      return R;
    }
    */

  public: // ---- Broadcasting -------------------------------------------------------------------------------

    
    void broadcast0(const Tensor& x){
      TimedFn T("Ptensorsf0","brcast0",*this,x);
      add(x);
    }

    void broadcast0(const Tensor& x, const int offs){
      TimedFn T("Ptensorsf0","brcast0",*this,x);
      cols(offs,offs+x.dims[1])+=x;
    }

    /*
    void broadcast0(const RtensorPackB& x, const AindexPack& list){
      TimedFn T("Ptensorsf0","brcast0",*this,x,list,list.size()*nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  view_of(list.tens(i),list.ix(i))+=x.view1_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf0_broadcast0_cu(*this,x,list,0,stream)));
    }

    void broadcast0(const RtensorPackB& x, const AindexPack& list, const int offs){
      TimedFn T("Ptensorsf0","brcast0",*this,x,list,list.size()*nc);
      if(dev==0){
	int N=list.size();
	const int n=x.nc;
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i),offs,n)+=x.view1_of(i);
      }
      GPUCODE(CUDA_STREAM(Ptensorsf0_broadcast0_cu(*this,x,list,offs,stream)));
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensorsf0";
    }

    string repr() const{
      return "<Ptensorsf0[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(dev>0){
	Ptensorsf0 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensorsf0& x){
      stream<<x.str(); return stream;}

  };

}


#endif 

