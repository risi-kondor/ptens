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
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "Ptensor1.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensorsb.hpp"
#include "AtomsPackTag.hpp"


namespace ptens{


  template<typename TYPE>
  class Ptensors1: public Ptensorsb<TYPE>, public cnine::diff_class<Ptensors1<TYPE> >{
  public:

    friend class Ptensors0<TYPE>;
    friend class Ptensors2<TYPE>;

    typedef Ptensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor2_view Rtensor2_view;

    using cnine::diff_class<Ptensors1<TYPE> >::grad;

    using BASE::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;
    using TENSOR::dev;
    using TENSOR::strides;
    using TENSOR::get_arr;
    using TENSOR::cols;


    AtomsPack atoms;
    AtomsPackTag1 tag;


    ~Ptensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1(){}

    //Ptensors1(const TENSOR& M):
    //BASE(M.copy()){} // for diff_class, unsafe!!

    Ptensors1(const TENSOR& M, const AtomsPack& _atoms):
      BASE(M),
      atoms(_atoms),
      tag(_atoms){}

    Ptensors1(const TENSOR& M, const AtomsPackTag1& _tag):
      BASE(M),
      atoms(_tag.obj->atoms.lock()),
      tag(_tag){}

    Ptensors1(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.nrows1(),nc),0,_dev),
      atoms(_atoms),
      tag(_atoms){}

    Ptensors1(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(cnine::Gdims(_atoms.nrows1(),nc),fcode,_dev),
      atoms(_atoms),
      tag(_atoms){}


    static Ptensors1 cat(const vector<Ptensors1>& list){
      vector<AtomsPack> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      if(ptens_global::cache_atomspack_cats) 
	return Ptensors1(cnine::Ltensor<TYPE>::stack(0,list),ptens_global::atomspack_cat_cache(v));
      return Ptensors1(cnine::Ltensor<TYPE>::stack(0,list),AtomsPack::cat(v));
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors1(const AtomsPack& _atoms, const Args&... args):
      atoms(_atoms),
      tag(_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.nrows1(),v.nc},v.fcode,v.dev));
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
    

    static Ptensors1 zero(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1(_atoms,nc,0,_dev);}

    static Ptensors1 raw(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1(_atoms,nc,1,_dev);}

    static Ptensors1 ones(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1(_atoms,nc,2,_dev);}

    static Ptensors1 sequential(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1(_atoms,nc,3,_dev);}

    static Ptensors1 gaussian(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensors1(_atoms,nc,4,_dev);}


  public: // ----- Spawning ----------------------------------------------------------------------------------

    
    Ptensors1 copy() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(TENSOR::copy(),atoms);
    }

    Ptensors1 copy(const int _dev) const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(TENSOR::copy(_dev),atoms);
    }

    Ptensors1 zeros_like() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(TENSOR::zeros_like(),atoms);
    }

    Ptensors1 zeros_like(const int nc) const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(TENSOR({dim(0),nc},0,get_dev()),atoms);
    }

    Ptensors1 gaussian_like() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(BASE::gaussian_like(),atoms);
    }

    static Ptensors1 zeros_like(const Ptensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors1 gaussian_like(const Ptensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors1(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors1* new_zeros_like(const Ptensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new Ptensors1(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    //Ptensors1(const Ptensors1& x):
    //BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
    //atoms(x.atoms){
    //BASE::view2().set(x.view_as_matrix().view2());
    //}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors1(const Ptensors1& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms),
      tag(x.tag){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors1& get_grad(){
      return cnine::diff_class<Ptensors1<TYPE> >::get_grad();
    }

    const Ptensors1& get_grad() const{
      return cnine::diff_class<Ptensors1<TYPE> >::get_grad();
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

    AtomsPack get_atoms() const{
      return atoms;
    }

    int size_of(const int i) const{
      return atoms.size_of(i);
    }

    int offset(const int i) const{
      return atoms.row_offset1(i);
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
    static Ptensors1<float> linmaps(const SOURCE& x){
      Ptensors1<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const Ptensors1<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      cols(nc,nc)+=x;
    }

    void add_linmaps(const Ptensors2<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      cols(2*nc,3*nc)+=x.reduce1();
    }

    void add_linmaps_back(const Ptensors0<TYPE>& r){
      broadcast0(r);
    }

    void add_linmaps_back(const Ptensors1<TYPE>& r){
      int nc=get_nc();
      broadcast0(r.reduce0(0,nc));
      add(r.cols(nc,nc));
    }

    void add_linmaps_back(const Ptensors2<TYPE>& r){
      int nc=get_nc();
      broadcast0(r.reduce0_shrink(0,nc));
      add(r.reduce1_shrink(2*nc,nc));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------

    
    template<typename SOURCE>
    static Ptensors1<TYPE> gather(const SOURCE& x, const AtomsPack& a, const int min_overlaps=1){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      Ptensors1<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,min_overlaps);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x, const int min_overlaps=1){
      auto overlaps=ptens_global::overlaps_cache(atoms,x.atoms);
      if(ptens_global::row_level_operations){
	rmap(x,overlaps)(*this,x);
      }else{
      }
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      auto overlaps=ptens_global::overlaps_cache(x.atoms,atoms);
      if(ptens_global::row_level_operations){
	x.rmap(*this,overlaps).inv()(*this,x);
      }else{
      }
    }

    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){
      auto overlaps=ptens_global::overlaps_cache(x.atoms,atoms);
      if(ptens_global::row_level_operations){
	x.rmap(*this,overlaps).inv()(get_grad(),x.get_grad());
      }else{
      }
    }


  private:

    template<typename SOURCE>
    RowLevelMap& rmap(const SOURCE& x, const shared_ptr<TensorLevelMapObj>& tmap) const{
      return *ptens_global::rmap_cache(tag,x.tag,tmap);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      TimedFn T("Ptensors1","reduce0",*this);
      int N=size();
      int dev=get_dev();
      BASE R({N,get_nc()},0,dev);
      Rtensor2_view r=R.view2();
      for(int i=0; i<N; i++)
	view2_of(i).sum0_into(r.slice0(i));
      return R;
    }

    BASE reduce0(const int offs, const int nc) const{
      TimedFn T("Ptensors1","reduce0",*this);
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
      TimedFn T("Ptensors1","broadcast0",*this);
      int N=size();
      int nc=X.dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      Rtensor2_view x=X.view2();
      
      for(int i=0; i<N; i++)
	view2_of(i,offs,nc)+=cnine::repeat0(x.slice0(i),size_of(i));
    }

    void broadcast1(const BASE& X, const int offs=0){
      TimedFn T("Ptensors1","broadcast1",*this);
      int nc=X.dim(1);
      BASE::view2().block(0,offs,dim(0),nc)+=X.view2();
    }


   public: // ---- Message passing ----------------------------------------------------------------------------


    

   public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors1";
    }

    string repr() const{
      return "<Ptensors1[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors1 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<(*this)(i).str(indent)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors1& x){
      stream<<x.str(); return stream;}



  };


  template<typename SOURCE>
  inline Ptensors1<float> linmaps1(const SOURCE& x){
    Ptensors1<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  Ptensors1<float> gather1(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
    Ptensors1<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }


}


#endif 

    //Ptensors1(const TENSOR& x, const shared_ptr<PtensorsJig1<int> >& _jig):
    //BASE(x),
    //atoms(_jig->atoms),
    //jig(_jig){}

    /*
    Ptensors1(const AtomsPack1& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors1(const AtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}
    */

