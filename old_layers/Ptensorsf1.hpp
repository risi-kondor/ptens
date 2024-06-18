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

#ifndef _ptens_Ptensorsf1
#define _ptens_Ptensorsf1

#include "Ptens_base.hpp"

#include "Tensor.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensorsf0.hpp"
#include "diff_class.hpp"

#include "PtensLoggedTimer.hpp"


namespace ptens{


  class Ptensorsf1: public cnine::Tensor<float>, public cnine::diff_class<Ptensorsf1>{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::IntTensor itensor;
    typedef cnine::IntTensor IntTensor;
    typedef cnine::Tensor<float> Tensor;
    //typedef cnine::RtensorPackB RtensorPack;
    //typedef cnine::Rtensor1_view Rtensor1_view;
    //typedef cnine::Rtensor2_view Rtensor2_view;
    //typedef cnine::Rtensor3_view Rtensor3_view;
    //typedef cnine::RtensorPackB RtensorPackB;

    //int nc; // duplicates the same variable in RtensorPackB
    Tensor<int> atoms;
    //bool is_view=false;


    ~Ptensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //Ptensorsf1(){}

    //Ptensorsf1(const int _nc, const int _dev=0):
    //Tensor(2,_nc,_dev) /*, nc(_nc)*/{}

    //Ptensorsf1(const AtomsPack& _atoms, const int _nc, const int _dev=0):
    //RtensorPackB(2,_nc,_dev), /*nc(_nc),*/ atoms(_atoms){
    //}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensorsf1(const int _n, const int _k, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      Tensor({_n,_k,_nc},dummy,_dev), atoms({_n,_k},cnine::fill_zero()){}


  public: // ----- Constructors ------------------------------------------------------------------------------


    static Ptensorsf1 raw(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_raw(),_dev);}

    static Ptensorsf1 zero(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_zero(),_dev);}

    static Ptensorsf1 gaussian(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensorsf1 gaussian(const int _n, const int _k, const int _nc, const float sigma, const int _dev){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensorsf1 randn(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensorsf1 randn(const int _n, const int _k, const int _nc, const float sigma, const int _dev){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensorsf1 sequential(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensorsf1(_n,_k,_nc,cnine::fill_sequential(),0);}


    /*
    static Ptensorsf1 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensorsf1 R(_atoms,_nc,_dev);
      R.reserve(_atoms.tsize1()*_nc);
      R.dir=IntTensor::raw({_atoms.size(),3});
      R.tail=0;
      for(int i=0; i<_atoms.size(); i++){
	R.dir.set_row(i,{R.tail,_atoms.size_of(i),_nc});
	R.tail+=_atoms.size_of(i)*_nc;
      }
      return R;
    }
    

    static Ptensorsf1 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensorsf1 R(_atoms,_nc,_dev);
      R.reserve_zero(_atoms.tsize1()*_nc);
      R.dir=IntTensor::raw({_atoms.size(),3});
      R.tail=0;
      for(int i=0; i<_atoms.size(); i++){
	R.dir.set_row(i,{R.tail,_atoms.size_of(i),_nc});
	R.tail+=_atoms.size_of(i)*_nc;
      }
      return R;
    }

    static Ptensorsf1 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensorsf1 R(_nc,0);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor1::gaussian(_atoms(i),_nc));
      R.to_device(_dev);
      return R;
    }

    static Ptensorsf1 gaussian(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      Ptensorsf1 R(_nc,0);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor1::gaussian(_atoms(i),_nc,sigma,0));
      R.to_device(_dev);
      return R;
    }

    static Ptensorsf1 randn(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensorsf1::gaussian(_atoms,_nc,_dev);
    }

    static Ptensorsf1 randn(const AtomsPack& _atoms, const int _nc, const float sigma, const int _dev){
      return Ptensorsf1::gaussian(_atoms,_nc,sigma,_dev);
    }

    static Ptensorsf1 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensorsf1 R(_nc,0);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor1::sequential(_atoms(i),_nc));
      R.to_device(_dev);
      return R;
    }


    static Ptensorsf1 concat(const Ptensorsf1& x, const Ptensorsf1& y){
      Ptensorsf1 R=Ptensorsf1::zero(x.atoms,x.nc+y.nc,x.dev);
      R.add_to_channels(x,0);
      R.add_to_channels(y,x.nc);
      return R;
    }
    */


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensorsf1(const Ptensorsf1& x):
      RtensorPackB(x),
      cnine::diff_class<Ptensorsf1>(x),
      atoms(x.atoms)
      /*,nc(x.nc)*/{
      PTENS_COPY_WARNING();
    }
	
    Ptensorsf1(Ptensorsf1&& x):
      Tensor(std::move(x)),
      cnine::diff_class<Ptensorsf1>(std::move(x)),
      atoms(std::move(x.atoms))
      /*,nc(x.nc)*/{
      PTENS_MOVE_WARNING();
    }

    Ptensorsf1& operator=(const Ptensorsf1& x)=delete;


  public: // ---- Spawning -----------------------------------------------------------------------------------


   static Ptensorsf1 zeros_like(const Ptensorsf1& x){
     return Ptensorsf1(Tensor::zeros_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf1* new_zeros_like(const Ptensorsf1& x){
      return new Ptensorsf1(Tensor::zeros_like(x),x.atoms);//,x.nc);
    }

   static Ptensorsf1 gaussian_like(const Ptensorsf1& x){
     return Ptensorsf1(Tensor::gaussian_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf1 randn_like(const Ptensorsf1& x){
      return Ptensorsf1(Tensor::gaussian_like(x),x.atoms);//,x.nc);
    }

    static Ptensorsf1 sequential_like(const Ptensorsf1& x){
      return Ptensorsf1(Tensor::sequential_like(x),x.atoms);//,x.nc);
    }

    
  public: // ----- Conversions -------------------------------------------------------------------------------


    //Ptensorsf1(cnine::RtensorPack&& x, const AtomsPack& _atoms)://, const int _nc):
      //RtensorPackB(std::move(x)), atoms(_atoms)/*, nc(_nc)*/{}

    Ptensorsf1(RtensorPackB&& x, const AtomsPack& _atoms):
      RtensorPackB(std::move(x)), atoms(_atoms)/*, nc(x.nc)*/{}

    //rtensor view_as_matrix() const{
    //return rtensor::view_of_blob({tail/nc,nc},get_arr(),dev);
    //}

    Ptensorsf1(const rtensor& A, const AtomsPack& _atoms):
      RtensorPackB(A,_atoms.dims1(A.dim(1))), atoms(_atoms){
      //nc=A.dim(1);
    }

    #ifdef _WITH_ATEN
    Ptensorsf1(const at::Tensor& T, const AtomsPack& _atoms):
      Ptensorsf1(rtensor::regular(T),_atoms){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensorsf1(const Ptensorsf1& x, const int _dev):
      RtensorPackB(x,_dev),
      atoms(x.atoms)/*,nc(x.nc)*/{}

    Ptensorsf1& to_device(const int _dev){
      RtensorPackB::to_device(_dev);
      return *this;
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    //int get_nc() const{
    //return nc;
    //}

    AtomsPack& get_atomsref(){
      return atoms;
    }

    AtomsPack view_of_atoms(){
      return atoms.view();
    }


    int size() const{
      return dims(0);
    }

    int getn() const{
      return dims(0);
    }

    int get_nc() const{
      return dims[2];
    }

    int getk() const{
      return dims[1];
    }

    /*
    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }

    rtensor tensor_of(const int i) const{
      return RtensorPackB::operator()(i);
    }

    Rtensor2_view view_of(const int i) const{
      return RtensorPackB::view2_of(i);
    }

    Rtensor2_view view_of(const int i, const int offs, const int n) const{
      return RtensorPackB::view2_of(i).block(0,offs,-1,n);
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
    */


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_to_channels(const Ptensorsf1& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i,offs,x.nc)+=x.view_of(i);
    }

    void add_channels(const Ptensorsf1& x, const int offs){
      PTENS_CPUONLY();
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i)+=x.view_of(i,offs,nc);
    }

    /*
    void add_mprod(const Ptensorsf1& x, const rtensor& y){
      PTENS_CPUONLY();
      PTENS_ASSRT(x.size()==size());
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i).add_matmul_AA(x.view_of(i),y.view2());
      }else{
	view_as_matrix().add_mprod(x.view_as_matrix(),y);
      }
    }

    void add_mprod_back0(const Ptensorsf1& g, const rtensor& y){
      PTENS_CPUONLY();
      PTENS_ASSRT(g.size()==size());
      if(dev==0){
	for(int i=0; i<size(); i++)
	  view_of(i).add_matmul_AT(g.view_of(i),y.view2());
      }else{
	view_as_matrix().add_Mprod_AT(g.view_as_matrix(),y);
      }
    }

    void add_mprod_back1_to(rtensor& r, const Ptensorsf1& x) const{
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

    Ptensorsf1 scale_channels(const rtensor& y) const{
      return Ptensorsf1(RtensorPackB::scale_channels(y.view1()),atoms);
    }

 
  public: // ---- Reductions ---------------------------------------------------------------------------------


    Tensor reduce0() const{
      TimedFn T("Ptensorsf1","reduce0",*this);
      Tensor R({getn(),getk(),get_nc()},cnine::fill_zero(),dev);
      R.add_sum(1,*this);
      return R;
    }

    Tensor reduce0_n() const{
      TimedFn T("Ptensorsf1","reduce0_n",*this);
      PTENS_UNIMPL();
      Tensor R({getn(),getk(),get_nc()},cnine::fill_zero(),dev);
      R.add_sum(1,*this);
      return R;
    }

    Tensor reduce0(const int offs, const int n) const{
      TimedFn T("Ptensorsf1","reduce0",*this);
      Tensor R({getn(),getk(),n},cnine::fill_zero(),dev);
      R.add_sum(1,slices(2,offs,offs+n));
      return R;
    }

    /*
    Tensor reduce0(const AindexPack& list) const{
      TimedFn T("Ptensorsf1","reduce0",*this,list,list.count1*nc);
      int N=list.size();
      Tensor R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).sum0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    Tensor reduce0_n(const AindexPack& list) const{
      TimedFn T("Ptensorsf1","reduce0_n",*this,list,list.count1*nc);
      int N=list.size();
      Tensor R(N,Gdims(nc),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).avg0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_reduce0n_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    Tensor reduce0(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensorsf1","reduce0",*this,list,list.count1*n);
      int N=list.size();
      Tensor R(N,Gdims(n),cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n).sum0_into(R.view1_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_reduce0_cu(R,*this,list,offs,n,stream)));
      return R;
    }
    */

    Tensor reduce1() const{
      TimedFn T("Ptensorsf1","reduce1",*this);
      return *this;
    }

    Tensor reduce1(const int offs, const int n) const{
      TimedFn T("Ptensorsf1","reduce1",*this);
      return slices(2,offs,offs+n);
    }

    /*
    Tensor reduce1(const AindexPack& list) const{
      TimedFn T("Ptensorsf1","reduce1",*this,list,list.count1*nc);
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),nc});
      Tensor R(dims,cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view2_of(i)+=view_of(list.tens(i),list.ix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_reduce1_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    Tensor reduce1(const AindexPack& list, const int offs, const int n) const{
      TimedFn T("Ptensorsf1","reduce1",*this,list,list.count1*n);
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),n});
      Tensor R(dims,cnine::fill_zero(),dev);
      if(dev==0){
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view2_of(i)+=view_of(list.tens(i),list.ix(i),offs,n);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_reduce1_cu(R,*this,list,offs,n,stream)));
      return R;
    }
    */


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const Tensor& x){
      TimedFn T("Ptensorsf1","brcast0",*this,x);
      add_broadcast(1,x);
    }

    void broadcast0_n(const Tensor& x){
      TimedFn T("Ptensorsf1","brcast0",*this,x);
      CNINE_UNIMPL();
      add_broadcast(1,x);
    }

    void broadcast0(const Tensor& x, const int offs){
      TimedFn T("Ptensorsf1","brcast0",*this,x);
      slices(2,offs,offs+x.dim(2)).add_broadcast(1,x);
    }

    /*
    void broadcast0(const Tensor& x, const AindexPack& list){
      TimedFn T("Ptensorsf1","brcast0",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i))+=repeat0(x.view1_of(i),list.nix(i));
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_broadcast0_cu(*this,x,list,0,stream)));
    }

    void broadcast0_n(const Tensor& x, const AindexPack& list){
      TimedFn T("Ptensorsf1","brcast0_n",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++)
	  view_of(list.tens(i),list.ix(i)).add(repeat0(x.view1_of(i),list.nix(i)),1.0/((float)list.nix(i))); // check this
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_broadcast0n_cu(*this,x,list,0,stream)));
    }

    void broadcast0(const Tensor& x, const AindexPack& list, const int offs){
      TimedFn T("Ptensorsf1","brcast0",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	const int n=x.nc;
	for(int i=0; i<N; i++){
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view1_of(i),list.nix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_broadcast0_cu(*this,x,list,offs,stream)));
      }
    */

    void broadcast1(const Tensor& x){
      TimedFn T("Ptensorsf1","brcast1",*this,x);
      add(x);
    }

    void broadcast1(const Tensor& x, const int offs){
      TimedFn T("Ptensorsf1","brcast1",*this,x);
      slices(2,offs,x.dim(2)).add(x);
    }

    /*
    void broadcast1(const Tensor& x, const AindexPack& list){
      TimedFn T("Ptensorsf1","brcast1",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.dim_of(i,0)==0) continue;
	  view_of(list.tens(i),list.ix(i))+=x.view2_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_broadcast1_cu(*this,x,list,0,stream)));
    }

    void broadcast1(const Tensor& x, const AindexPack& list, const int offs){
      TimedFn T("Ptensorsf1","brcast1",*this,x,list,list.count1*x.nc);
      if(dev==0){
	int N=list.size();
	const int n=x.nc;
	for(int i=0; i<N; i++){
	  if(x.dim_of(i,0)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n)+=x.view2_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensorsf1_broadcast1_cu(*this,x,list,offs,stream)));
    }
    */



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensorsf1";
    }

    string repr() const{
      if(dev==0) return "<Ptensorsf1[N="+to_string(size())+"]>";
      else return "<Ptensorsf1[N="+to_string(size())+"][G]>";
    }

    string str(const string indent="") const{
      if(dev>0){
	Ptensorsf1 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
	//oss<<indent<<"Ptensor "<<i<<" "<<Atoms(atoms(i))<<":"<<endl;
	//oss<<Tensor::operator()(i).str()<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensorsf1& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
