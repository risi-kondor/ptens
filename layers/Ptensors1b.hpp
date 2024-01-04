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

#ifndef _ptens_Ptensors1b
#define _ptens_Ptensors1b

#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "AtomsPack1.hpp"
#include "Ptensors1.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensorsb.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors0b;
  template<typename TYPE> class Ptensors2b;


  template<typename TYPE>
  class Ptensors1b: public Ptensorsb<TYPE>, public cnine::diff_class<Ptensors1b<TYPE> >{
  public:

    typedef Ptensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor2_view Rtensor2_view;

    using cnine::diff_class<Ptensors1b<TYPE> >::grad;

    using BASE::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;
    using TENSOR::dev;
    using TENSOR::strides;
    using TENSOR::get_arr;
    using TENSOR::cols;


    AtomsPack1 atoms;


    ~Ptensors1b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1b(const TENSOR& M):
      BASE(M.copy()){} // for diff_class

    Ptensors1b(const AtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors1b(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.tsize1(),nc),0,_dev),
      atoms(_atoms){}

    Ptensors1b(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(cnine::Gdims(_atoms.tsize1(),nc),fcode,_dev),
      atoms(_atoms){}


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors1b(const AtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.size1(),v.nc},v.fcode,v.dev));
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

    
    Ptensors1b copy() const{
      return Ptensors1b(TENSOR::copy(),atoms);
    }

    Ptensors1b copy(const int _dev) const{
      return Ptensors1b(TENSOR::copy(_dev),atoms);
    }

    Ptensors1b zeros_like() const{
      return Ptensors1b(TENSOR::zeros_like(),atoms);
    }

    static Ptensors1b* new_zeros_like(const Ptensors1b& x){
      return new Ptensors1b(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors1b(const TENSOR& x, const AtomsPack1& _atoms):
      BASE(x),
      atoms(_atoms){}

    Ptensors1b(const Ptensors1& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors1b(const Ptensors1b& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors1b& get_grad(){
      return cnine::diff_class<Ptensors1b<TYPE> >::get_grad();
    }

    const Ptensors1b& get_grad() const{
      return cnine::diff_class<Ptensors1b<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 1;
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

    int offset(const int i) const{
      return atoms.offset(i);
    }

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    Rtensor2_view view2_of(const int i) const{
      return Rtensor2_view(get_arr()+offset(i)*strides[0],size_of(i),get_nc(),strides[0],strides[1],dev);
    }

    Rtensor2_view view2_of(const int i, const int offs, const int m) const{
      return Rtensor2_view(get_arr()+offset(i)*strides[0]+offs*strides[1],
	size_of(i),m,strides[0],strides[1],dev);
    }

    TENSOR tensor_of(const int i) const{
      return TENSOR::rows(offset(i),size_of(i));
    }

    Ptensor1 operator()(const int i) const{
      return Ptensor1(cnine::RtensorA(tensor_of(i).view2()),atoms_of(i));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE>
    static Ptensors1b<TYPE> gather(const SOURCE& x, const AtomsPack& a){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      Ptensors1b<TYPE> R(a,nc,x.get_dev());
      R.gather(x);
      return R;
    }

    void add_linmaps(const Ptensorsb<TYPE>& x){
      int xk=x.getk();
      int nc=get_nc();
      if(xk==0) 
	broadcast0(x);
      if(xk==1){
	broadcast0(x.reduce0());
	cols(nc,nc)+=x;}
      if(xk==2){
	broadcast0(x.reduce0());
	cols(2*nc,3*nc)+=x.reduce1();}
    }

    void add_linmaps_back(const Ptensorsb<TYPE>& r){
      int k=r.getk();
      int nc=get_nc();
      int nc_out=vector<int>({1,2,5})[k]*nc;
      PTENS_ASSRT(r.get_nc()==nc_out);
      if(k==0) 
	broadcast0(r);
      if(k==1){
	broadcast0(r.reduce0(0,nc));
	add(r.cols(nc,nc));}
      if(k==2){
	broadcast0(r.reduce0_shrink(0,nc));
	add(r.reduce1_shrink(2*nc,nc));}
    }

    template<typename SOURCE>
    void gather(const SOURCE& x){
      (atoms.overlaps_mmap(x.atoms))(*this,x);
    }

    template<typename OUTPUT>
    void gather_back(const OUTPUT& x){
      x.atoms.overlaps_mmap(atoms).inv()(*this,x);
    }

    template<typename OUTPUT>
    void gather_backprop(const OUTPUT& x){
      get_grad().gather_back(x.get_grad());
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      TimedFn T("Ptensors1b","reduce0",*this);
      int N=size();
      int dev=get_dev();
      BASE R({N,get_nc()},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++)
	  view2_of(i).sum0_into(r.slice0(i));
      }
    }

    BASE reduce1() const{
      return *this;
    }

    
  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      TimedFn T("Ptensors1b","broadcast0",*this);
      int N=size();
      PTENS_ASSRT(X.dim(0)==N);
      Rtensor2_view x=X.view2();
      
      if(get_dev()==0){
	for(int i=0; i<N; i++)
	  view2_of(i,offs,get_nc())+=cnine::repeat0(x.slice0(i),size_of(i));
      }
    }

    void broadcast1(const BASE& X, const int offs=0){
      TimedFn T("Ptensors1b","broadcast1",*this);
      BASE::view2().block(0,offs,dim(0),get_nc())+=X.view2();
    }


   public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors1b";
    }

    string repr() const{
      return "<Ptensors1b[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors1b y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors1b& x){
      stream<<x.str(); return stream;}



  };


}


#endif 

    /*
    static Ptensors1b<TYPE> gather(const Ptensors0b<TYPE>& x, const AtomsPack& a){
      Ptensors1b<TYPE> R(a,x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static Ptensors1b<TYPE> gather(const Ptensors1b<TYPE>& x, const AtomsPack& a){
      Ptensors1b<TYPE> R(a,2*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static Ptensors1b<TYPE> gather(const Ptensors2b<TYPE>& x, const AtomsPack& a){
      Ptensors1b<TYPE> R(a,5*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }
    */

