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

#ifndef _ptens_Ptensors2b
#define _ptens_Ptensors2b

#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "AtomsPack2.hpp"
#include "Ptensor2.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensorsb.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors0b;
  template<typename TYPE> class Ptensors1b;


  template<typename TYPE>
  class Ptensors2b: public Ptensorsb<TYPE>, public cnine::diff_class<Ptensors2b<TYPE> >{
  public:

    typedef Ptensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    using cnine::diff_class<Ptensors2b<TYPE> >::grad;
    using BASE::get_dev;
    using TENSOR::dev;
    using TENSOR::get_arr;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::strides;
    using TENSOR::cols;
    using TENSOR::add;


    AtomsPack2 atoms;


    ~Ptensors2b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors2b(const TENSOR& M):
      BASE(M.copy()){} // for diff_class

    Ptensors2b(const AtomsPack2& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors2b(const AtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors2b(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.tsize2(),nc),0,_dev),
      atoms(_atoms){}

    Ptensors2b(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(cnine::Gdims(_atoms.tsize2(),nc),fcode,_dev),
      atoms(_atoms){}

    static Ptensors2b cat(const vector<Ptensors2b>& list){
      vector<AtomsPack2> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return Ptensors2b(AtomsPack2::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors2b(const AtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.size2(),v.nc},v.fcode,v.dev));
    }

    template<typename... Args>
    void unroller(vparams& v, const cnine::ChannelsArgument& x, const Args&... args){
      v.nc=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::FillArgument& x, const Args&... args){
      v.fcode=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DeviceArgument& x, const Args&... args){
      v.dev=x.get(); unroller(v, args...);}

    void unroller(vparams& v){}


  public: // ----- Spawning ----------------------------------------------------------------------------------


    Ptensors2b copy() const{
      return Ptensors2b(TENSOR::copy(),atoms);
    }

    Ptensors2b copy(const int _dev) const{
      return Ptensors2b(TENSOR::copy(_dev),atoms);
    }

    Ptensors2b zeros_like() const{
      return Ptensors2b(TENSOR::zeros_like(),atoms);
    }

    Ptensors2b gaussian_like() const{
      return Ptensors2b(BASE::gaussian_like(),atoms);
    }

    static Ptensors2b zeros_like(const Ptensors2b& x){
      return Ptensors2b(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors2b gaussian_like(const Ptensors2b& x){
      return Ptensors2b(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors2b* new_zeros_like(const Ptensors2b& x){
      return new Ptensors2b(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors2b(const TENSOR& x, const AtomsPack2& _atoms):
      BASE(x),
      atoms(_atoms){}

    //Ptensors2b(const Ptensors2& x):
    //BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
    //atoms(x.atoms){
    //BASE::view2().set(x.view_as_matrix().view2());
    //}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors2b(const Ptensors2b& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors2b& get_grad(){
      return cnine::diff_class<Ptensors2b<TYPE> >::get_grad();
    }

    const Ptensors2b& get_grad() const{
      return cnine::diff_class<Ptensors2b<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 2;
    }

    int size() const{
      return atoms.size();
    }

    int get_nc() const{
      return BASE::dim(1);
    }

    int nchannels() const{
      return BASE::dim(1);
    }

    int size_of(const int i) const{
      return atoms.size_of(i);
    }

    AtomsPack get_atoms() const{
      return atoms.obj->atoms;
    }

    int offset(const int i) const{
      return atoms.offset(i);
    }

    int offset1(const int i) const{
      return atoms.offset1(i);
    }

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    TENSOR tensor_of(const int i) const{
      int k=size_of(i);
      return TENSOR::rows(offset(i),k*k).reshape({k,k,nchannels()});
    }

    Ptensor2<TYPE> operator()(const int i) const{
      return Ptensor2(tensor_of(i).view3(),atoms_of(i));
    }

    Rtensor3_view view3_of(const int i) const{
      int n=size_of(i);
      return Rtensor3_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0],n,n,get_nc(),strides[0]*n,strides[0],strides[1],dev);
    }

    Rtensor3_view view3_of(const int i, const int offs, const int m) const{
      int n=size_of(i);
      return Rtensor3_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0]+offs*strides[1],
	n,n,m,strides[0]*n,strides[0],strides[1],dev);
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE>
    static Ptensors2b<float> linmaps(const SOURCE& x){
      Ptensors2b<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE>
    static Ptensors2b<TYPE> gather(const SOURCE& x, const AtomsPack& a){
      int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
      Ptensors2b<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x);
      return R;
    }

    void add_linmaps(const Ptensors0b<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const Ptensors1b<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      broadcast1(x,2*nc);
    }

    void add_linmaps(const Ptensors2b<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      broadcast1(x.reduce1(),4*nc);
      broadcast2(x,13*nc);
    }


    void add_linmaps_back(const Ptensors0b<TYPE>& r){
      broadcast0_shrink(r);
    }

    void add_linmaps_back(const Ptensors1b<TYPE>& r){
      int nc=get_nc();
      broadcast0_shrink(r.reduce0(0,2*nc));
      broadcast1_shrink(r.cols(2*nc,3*nc));
    }

    void add_linmaps_back(const Ptensors2b<TYPE>& r){
      int nc=get_nc();
      broadcast0_shrink(r.reduce0_shrink(0,nc));
      broadcast1_shrink(r.reduce1_shrink(2*nc,9*nc));
      // TODO
    }


    template<typename SOURCE>
    void add_gather(const SOURCE& x){
      (atoms.overlaps_mmap(x.atoms))(*this,x);
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      x.atoms.overlaps_mmap(atoms).inv()(*this,x);
    }

    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){ // TODO
      x.atoms.overlaps_mmap(atoms).inv()(this->get_grad(),x.get_grad());
    }

    //template<typename OUTPUT>
    //void gather_backprop(const OUTPUT& x){
    //get_grad().gather_back(x.get_grad());
    //}


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      TimedFn T("Ptensors2b","reduce0",*this);
      int N=size();
      int nc=get_nc();
      int dev=get_dev();
      
      BASE R({N,2*nc},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	Rtensor2_view r0=R.block(0,0,N,nc);
	Rtensor2_view r1=R.block(0,nc,N,nc);
	for(int i=0; i<N; i++){
	  view3_of(i).sum01_into(r0.slice0(i));
	  view3_of(i).diag01().sum0_into(r1.slice0(i));
	}
      }
      return R;
    }


    BASE reduce0_shrink(const int offs, const int nc) const{
      TimedFn T("Ptensors2b","reduce0_shrink",*this);
      int N=size();
      int dev=get_dev();
      
      BASE R({N,nc},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  view3_of(i,offs,nc).sum01_into(r.slice0(i));
	  view3_of(i,offs+nc,nc).diag01().sum0_into(r.slice0(i));
	}
      }
      return R;
    }


    BASE reduce1() const{
      TimedFn T("Ptensors2b","reduce1",*this);
      int N=size();
      int nc=get_nc();
      int dev=get_dev();

      BASE R({atoms.size1(),3*nc},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i).sum0_into(r.block(roffs,0,n,nc));
	  view3_of(i).sum1_into(r.block(roffs,nc,n,nc));
	  r.block(roffs,2*nc,n,nc)+=view3_of(i).diag01();
	}
      }
      return R;
    }

    
    BASE reduce1_shrink(const int offs, const int nc) const{
      TimedFn T("Ptensors2b","reduce1_shrink",*this);
      int N=size();
      int dev=get_dev();

      BASE R({atoms.size1(),nc},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc).sum0_into(r.block(roffs,0,n,nc));
	  view3_of(i,offs+nc,nc).sum1_into(r.block(roffs,0,n,nc));
	  r.block(roffs,0,n,nc)+=view3_of(i,offs+2*nc,nc).diag01();
	}
      }
      return R;
    }


    BASE reduce2_shrink(const int offs, const int nc) const{
      TimedFn T("Ptensors2b","reduce2_shrink",*this);
      int N=size();
      int dev=get_dev();
      
      BASE R({dim(0),nc},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset(i);
	  int n=size_of(i);
	  r.block(roffs,0,n*n,nc)+=view3_of(i,offs,nc);
	  r.block(roffs,0,n*n,nc)+=view3_of(i,offs+nc,nc).transp01();
	}
      }
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      TimedFn T("Ptensors2b","broadcast0",*this);
      int N=size();
      int dev=get_dev();
      int nc=X.dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      Rtensor2_view x=X.view2();
      
      if(dev==0){
	for(int i=0; i<N; i++){
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=repeat0(repeat0(x.slice0(i),n),n);
	  view3_of(i,offs+nc,nc).diag01()+=repeat0(x.slice0(i),n);
	}
      }
    }

    void broadcast0_shrink(const BASE& X){
      TimedFn T("Ptensors2b","broadcast0_shrink",*this);
      int N=size();
      int dev=get_dev();
      int nc=dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      PTENS_ASSRT(X.dim(1)==2*nc);
      Rtensor2_view x=X.view2();
      Rtensor2_view x0=x.block(0,0,N,nc);
      Rtensor2_view x1=x.block(0,nc,N,nc);
      
      if(dev==0){
	for(int i=0; i<N; i++){
	  int n=size_of(i);
	  view3_of(i)+=repeat0(repeat0(x0.slice0(i),n),n);
	  view3_of(i).diag01()+=repeat0(x1.slice0(i),n);
	}
      }
    }
    

    void broadcast1(const BASE& X, const int offs=0){
      TimedFn T("Ptensors2b","broadcast1",*this);
      int N=size();
      int dev=get_dev();
      int nc=X.dim(1);
      Rtensor2_view x=X.view2();

      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=repeat0(x.block(roffs,0,n,nc),n);
	  view3_of(i,offs+nc,nc)+=repeat1(x.block(roffs,0,n,nc),n);
	  view3_of(i,offs+2*nc,nc).diag01()+=x.block(roffs,0,n,nc);
	}
      }
    }


    void broadcast1_shrink(const BASE& X){
      TimedFn T("Ptensors2b","broadcast1_shrink",*this);
      int N=size();
      int dev=get_dev();
      int nc=dim(1);
      PTENS_ASSRT(X.dim(1)==3*nc);
      Rtensor2_view x=X.view2();
      Rtensor2_view x0=x.block(0,0,X.dim(0),nc);
      Rtensor2_view x1=x.block(0,nc,X.dim(0),nc);
      Rtensor2_view x2=x.block(0,2*nc,X.dim(0),nc);
      

      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i)+=repeat0(x.block(roffs,0,n,nc),n);
	  view3_of(i)+=repeat1(x.block(roffs,nc,n,nc),n);
	  view3_of(i).diag01()+=x.block(roffs,2*nc,n,nc);
	}
      }
    }


    void broadcast2(const BASE& X, const int offs=0){
      TimedFn T("Ptensors2b","broadcast2",*this);
      int N=size();
      int dev=get_dev();
      int nc=X.dim(1);
      Rtensor2_view x=X.view2();
      
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=split0(x.block(roffs,0,n*n,nc),n,n);
	  view3_of(i,offs+nc,nc)+=split0(x.block(roffs,0,n*n,nc),n,n).transp01();
	}
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors2b";
    }

    string repr() const{
      return "<Ptensors2b[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors2b y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors2b& x){
      stream<<x.str(); return stream;}



  };


  template<typename SOURCE>
  inline Ptensors2b<float> linmaps2(const SOURCE& x){
    Ptensors2b<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  Ptensors2b<float> gather2(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
    Ptensors2b<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }

}


#endif 

    /*
    static Ptensors2b<TYPE> gather(const Ptensors0b<TYPE>& x, const AtomsPack& a){
      Ptensors2b<TYPE> R(a,2*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static Ptensors2b<TYPE> gather(const Ptensors1b<TYPE>& x, const AtomsPack& a){
      Ptensors2b<TYPE> R(a,5*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static Ptensors2b<TYPE> gather(const Ptensors2b<TYPE>& x, const AtomsPack& a){
      Ptensors2b<TYPE> R(a,15*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }
    */
