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
#include "AtomsPack1.hpp"
#include "Ptensors1.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors0b;
  template<typename TYPE> class Ptensors2b;


  template<typename TYPE>
  class Ptensors1b: public cnine::Ltensor<TYPE>, public cnine::diff_class<Ptensors1b<TYPE> >{
  public:

    using cnine::diff_class<Ptensors1b<TYPE> >::grad;
    using cnine::diff_class<Ptensors1b<TYPE> >::get_grad;

    typedef cnine::Ltensor<TYPE> BASE;
    using BASE::get_dev;


    AtomsPack1 atoms;


    ~Ptensors1b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1b(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.tsize1(),nc),0,_dev),
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


    static Ptensors1b* new_zeros_like(const Ptensors1b& x){
      return new Ptensors1b(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors1b(BASE& x, const AtomsPack1& _atoms):
      BASE(x),
      atoms(_atoms){}

    Ptensors1b(const Ptensors1& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    #ifdef _WITH_ATEN
    Ptensors1b(const at::Tensor& T):
      BASE(T){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors1b(const Ptensors1b& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Access ------------------------------------------------------------------------------------


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
    
    BASE tensor_of(const int i) const{
      return BASE::rows(offset(i),offset(i)+size_of(i));
    }

    Ptensor1 operator()(const int i) const{
      return Ptensor1(cnine::RtensorA(tensor_of(i).view2()),atoms_of(i));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


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

