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

#ifndef _ptens_Ptensors0b
#define _ptens_Ptensors0b

#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "AtomsPack0.hpp"
#include "Ptensors0.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensorsb.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors1b;
  template<typename TYPE> class Ptensors2b;


  template<typename TYPE>
  class Ptensors0b: public Ptensorsb<TYPE>, public cnine::diff_class<Ptensors0b<TYPE> >{

  public:

    typedef Ptensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    using cnine::diff_class<Ptensors0b<TYPE> >::grad;
    using TENSOR::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;


    AtomsPack0 atoms;


    ~Ptensors0b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0b(){}

    Ptensors0b(const TENSOR& M): 
      BASE(M.copy()),
      atoms(AtomsPack0(M.dim(0))){}

    Ptensors0b(const AtomsPack0& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors0b(const AtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors0b(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.size(),nc),0,_dev),
      atoms(_atoms){}

    Ptensors0b(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(cnine::Gdims(_atoms.size(),nc),fcode,_dev),
      atoms(_atoms){}

    Ptensors0b(const int n, const int nc, const int fcode=0, const int _dev=0):
      BASE(cnine::Gdims(n,nc),fcode,_dev),
      atoms(n){}

    static Ptensors0b cat(const vector<Ptensors0b>& list){
      vector<AtomsPack0> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return Ptensors0b(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors0b(const AtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
      TENSOR::reset(TENSOR({atoms.size(),v.nc},v.fcode,v.dev));
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


    Ptensors0b copy() const{
      return Ptensors0b(TENSOR::copy(),atoms);
    }

    Ptensors0b copy(const int _dev) const{
      return Ptensors0b(TENSOR::copy(_dev),atoms);
    }

    Ptensors0b zeros_like() const{
      return Ptensors0b(TENSOR::zeros_like(),atoms);
    }

    Ptensors0b gaussian_like() const{
      return Ptensors0b(BASE::gaussian_like(),atoms);
    }

    static Ptensors0b zeros_like(const Ptensors0b& x){
      return Ptensors0b(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors0b zeros_like(const Ptensors0b& x, const int nc){
      return Ptensors0b(TENSOR({x.dim(0),nc},0,x.get_dev()),x.atoms);
    }

    static Ptensors0b gaussian_like(const Ptensors0b& x){
      return Ptensors0b(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors0b* new_zeros_like(const Ptensors0b& x){
      return new Ptensors0b(x.TENSOR::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors0b(const TENSOR& x, const AtomsPack0& _atoms):
      BASE(x),
      atoms(_atoms){}
    
    Ptensors0b(const Ptensors0& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(x.atoms){
      TENSOR::view2().set(x.view_as_matrix().view2());
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0b(const Ptensors0b& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors0b& get_grad(){
      return cnine::diff_class<Ptensors0b<TYPE> >::get_grad();
    }

    const Ptensors0b& get_grad() const{
      return cnine::diff_class<Ptensors0b<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 0;
    }

    int size() const{
      return atoms.size();
    }

    int get_nc() const{
      return TENSOR::dim(1);
    }

    int nchannels() const{
      return TENSOR::dim(1);
    }

    AtomsPack get_atoms() const{
      return atoms.obj->atoms;
    }

    int offset(const int i) const{
      return i;
    }

    //int nrows(const int i) const{
    //return 1;
    //}

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    TENSOR tensor_of(const int i) const{
      return TENSOR::row(offset(i));
    }

    Ptensor0 operator()(const int i) const{
      return Ptensor0(cnine::RtensorA(tensor_of(i).view1()),atoms_of(i));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0b<float> linmaps(const SOURCE& x){
      Ptensors0b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0b<TYPE> gather(const SOURCE& x, const AtomsPack& a){
      int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
      Ptensors0b<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x);
      return R;
    }

    void add_linmaps(const Ptensors0b<TYPE>& x){
      add(x);
    }

    void add_linmaps(const Ptensors1b<TYPE>& x){
      add(x.reduce0());
    }

    void add_linmaps(const Ptensors2b<TYPE>& x){
      add(x.reduce0());
    }

    void add_linmaps_back(const Ptensors0b<TYPE>& r){
      add(r);
    }

    void add_linmaps_back(const Ptensors1b<TYPE>& r){
      add(r.reduce0());
    }

    void add_linmaps_back(const Ptensors2b<TYPE>& r){
      int nc=get_nc();
      add(r.reduce0_shrink(0,nc));
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
    //x.atoms.overlaps_mmap(atoms).inv()(get_grad(),x.get_grad());
    //}


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      return *this;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      TimedFn T("Ptensors0b","broadcast0",*this);
      int nc=X.dim(1);
      BASE::view2().cols(offs,nc)+=X.view2();
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors0b";
    }

    string repr() const{
      return "<Ptensors0b[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors0b y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors0b& x){
      stream<<x.str(); return stream;}


  };


  template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
  inline Ptensors0b<float> linmaps0(const SOURCE& x){
    Ptensors0b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
  Ptensors0b<float> gather0(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
    Ptensors0b<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }



}


#endif 


    /*
    Ptensors0b(const at::Tensor& T):
      BASE(T){
      atoms=AtomsPack0(dim(0));
    }

    Ptensors0b(const at::Tensor& T, const AtomsPack& _atoms):
      BASE(T), atoms(_atoms){}

    Ptensors0b(const at::Tensor& T, const vector<vector<int> >& v):
      BASE(T), atoms(v){}
    */
    /*
    static Ptensors0b<TYPE> gather(const Ptensors0b<TYPE>& x, const AtomsPack& a){
      Ptensors0b<TYPE> R(a,x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static Ptensors0b<TYPE> gather(const Ptensors1b<TYPE>& x, const AtomsPack& a){
      Ptensors0b<TYPE> R(a,x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static Ptensors0b<TYPE> gather(const Ptensors2b<TYPE>& x, const AtomsPack& a){
      Ptensors0b<TYPE> R(a,2*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }
    */
