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
#include "Ptensor1.hpp"
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


    Ptensors1b(){}

    Ptensors1b(const TENSOR& M):
      BASE(M.copy()){} // for diff_class, unsafe!!

    Ptensors1b(const AtomsPack1& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors1b(const AtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors1b(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.tsize1(),nc),0,_dev),
      atoms(_atoms){}

    Ptensors1b(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(cnine::Gdims(_atoms.tsize1(),nc),fcode,_dev),
      atoms(_atoms){}

    static Ptensors1b cat(const vector<Ptensors1b>& list){
      vector<AtomsPack1> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return Ptensors1b(cnine::Ltensor<TYPE>::stack(0,list),AtomsPack1::cat(v));
    }


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


  public: // ---- Old style constructors ---------------------------------------------------------------------
    

    static Ptensors1b zero(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1b(_atoms,nc,0,_dev);}

    static Ptensors1b raw(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1b(_atoms,nc,1,_dev);}

    static Ptensors1b ones(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1b(_atoms,nc,2,_dev);}

    static Ptensors1b sequential(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1b(_atoms,nc,3,_dev);}

    static Ptensors1b gaussian(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1b(_atoms,nc,4,_dev);}


  public: // ----- Spawning ----------------------------------------------------------------------------------

    
    Ptensors1b copy() const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(TENSOR::copy(),atoms);
    }

    Ptensors1b copy(const int _dev) const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(TENSOR::copy(_dev),atoms);
    }

    Ptensors1b zeros_like() const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(TENSOR::zeros_like(),atoms);
    }

    Ptensors1b zeros_like(const int nc) const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(TENSOR({dim(0),nc},0,get_dev()),atoms);
    }

    Ptensors1b gaussian_like() const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(BASE::gaussian_like(),atoms);
    }

    static Ptensors1b zeros_like(const Ptensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors1b gaussian_like(const Ptensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1b(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors1b* new_zeros_like(const Ptensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new Ptensors1b(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors1b(const TENSOR& x, const AtomsPack1& _atoms):
      BASE(x),
      atoms(_atoms){}

    //Ptensors1b(const Ptensors1& x):
    //BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
    //atoms(x.atoms){
    //BASE::view2().set(x.view_as_matrix().view2());
    //}


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

    AtomsPack get_atoms() const{
      return atoms.obj->atoms;
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
      return Rtensor2_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0],size_of(i),get_nc(),strides[0],strides[1],dev);
    }

    Rtensor2_view view2_of(const int i, const int offs, const int m) const{
      return Rtensor2_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0]+offs*strides[1],
	size_of(i),m,strides[0],strides[1],dev);
    }

    TENSOR tensor_of(const int i) const{
      return TENSOR::rows(offset(i),size_of(i));
    }

    Ptensor1<TYPE> operator()(const int i) const{
      return Ptensor1(tensor_of(i),atoms_of(i));
    }

    const cnine::Rtensor3_view view3(const int K) const{
      int nc=get_nc();
      return cnine::Rtensor3_view(const_cast<float*>(get_arr()),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
    }

    cnine::Rtensor3_view view3(const int K){
      int nc=get_nc();
      return cnine::Rtensor3_view(get_arr(),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
    }

    const cnine::Rtensor3_view view3(const int K, const int offs, const int nc) const{
      int _nc=get_nc();
      return cnine::Rtensor3_view(const_cast<float*>(get_arr())+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
    }

    cnine::Rtensor3_view view3(const int K, const int offs, const int nc){
      int _nc=get_nc();
      return cnine::Rtensor3_view(get_arr()+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Linmaps ----------------------------------------------------------------------------


    template<typename SOURCE>
    static Ptensors1b<float> linmaps(const SOURCE& x){
      Ptensors1b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0b<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const Ptensors1b<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      cols(nc,nc)+=x;
    }

    void add_linmaps(const Ptensors2b<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      cols(2*nc,3*nc)+=x.reduce1();
    }

    void add_linmaps_back(const Ptensors0b<TYPE>& r){
      broadcast0(r);
    }

    void add_linmaps_back(const Ptensors1b<TYPE>& r){
      int nc=get_nc();
      broadcast0(r.reduce0(0,nc));
      add(r.cols(nc,nc));
    }

    void add_linmaps_back(const Ptensors2b<TYPE>& r){
      int nc=get_nc();
      broadcast0(r.reduce0_shrink(0,nc));
      add(r.reduce1_shrink(2*nc,nc));
    }

  public: // ---- Message passing ----------------------------------------------------------------------------

    
    template<typename SOURCE>
    static Ptensors1b<TYPE> gather(const SOURCE& x, const AtomsPack& a, const int min_overlaps=1){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      Ptensors1b<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,min_overlaps);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x, const int min_overlaps=1){
      (atoms.overlaps_mmap(x.atoms,min_overlaps))(*this,x);
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
      TimedFn T("Ptensors1b","reduce0",*this);
      int N=size();
      int dev=get_dev();
      BASE R({N,get_nc()},0,dev);
      Rtensor2_view r=R.view2();
      for(int i=0; i<N; i++)
	view2_of(i).sum0_into(r.slice0(i));
      return R;
    }

    BASE reduce0(const int offs, const int nc) const{
      TimedFn T("Ptensors1b","reduce0",*this);
      int N=size();
      int dev=get_dev();
      BASE R({N,nc},0,dev);
      Rtensor2_view r=R.view2();
      for(int i=0; i<N; i++)
	view2_of(i,offs,nc).sum0_into(r.slice0(i));
      return R;
    }

    BASE reduce1() const{
      return *this;
    }

    
  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      TimedFn T("Ptensors1b","broadcast0",*this);
      int N=size();
      int nc=X.dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      Rtensor2_view x=X.view2();
      
      for(int i=0; i<N; i++)
	view2_of(i,offs,nc)+=cnine::repeat0(x.slice0(i),size_of(i));
    }

    void broadcast1(const BASE& X, const int offs=0){
      TimedFn T("Ptensors1b","broadcast1",*this);
      int nc=X.dim(1);
      BASE::view2().block(0,offs,dim(0),nc)+=X.view2();
    }


   public: // ---- Message passing ----------------------------------------------------------------------------


    

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
	oss<<(*this)(i).str(indent)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors1b& x){
      stream<<x.str(); return stream;}



  };


  template<typename SOURCE>
  inline Ptensors1b<float> linmaps1(const SOURCE& x){
    Ptensors1b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  Ptensors1b<float> gather1(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
    Ptensors1b<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }


}


#endif 

