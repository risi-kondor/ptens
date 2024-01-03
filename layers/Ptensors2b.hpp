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
#include "AtomsPack2.hpp"
#include "Ptensors2.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensorsb.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors0b;
  template<typename TYPE> class Ptensors1b;


  template<typename TYPE>
  class Ptensors2b: public Ptensorsb<TYPE, Ptensors2b<TYPE> >, public cnine::diff_class<Ptensors2b<TYPE> >{
  public:

    using cnine::diff_class<Ptensors2b<TYPE> >::grad;
    using cnine::diff_class<Ptensors2b<TYPE> >::get_grad;

    typedef Ptensorsb<TYPE, Ptensors2b<TYPE> > BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    using BASE::get_dev;


    AtomsPack2 atoms;


    ~Ptensors2b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors2b(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.tsize2(),nc),0,_dev),
      atoms(_atoms){}


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


    //Ptensors2b like(const TENSOR& x) const{
    //return Ptensors2b(x,atoms);
    //}

    static Ptensors2b* new_zeros_like(const Ptensors2b& x){
      return new Ptensors2b(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors2b(const TENSOR& x, const AtomsPack2& _atoms):
      BASE(x),
      atoms(_atoms){}

    Ptensors2b(const Ptensors2& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    #ifdef _WITH_ATEN
    Ptensors2b(const at::Tensor& T):
      BASE(T){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors2b(const Ptensors2b& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors2b& get_grad(){
      return cnine::diff_class<Ptensors2b<TYPE> >::get_grad();
    }

    Ptensors2b& get_grad() const{
      return cnine::diff_class<Ptensors2b<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 2;
    }

    int size() const{
      return atoms.size();
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
    
    TENSOR tensor_of(const int i) const{
      int k=size_of(i);
      return TENSOR::rows(offset(i),k*k).reshape({k,k,nchannels()});
    }

    Ptensor2 operator()(const int i) const{
      return Ptensor2(cnine::RtensorA(tensor_of(i).view3()),atoms_of(i));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


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


}


#endif 

