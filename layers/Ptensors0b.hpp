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
#include "AtomsPack0.hpp"
#include "Ptensors0.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors1b;
  template<typename TYPE> class Ptensors2b;


  template<typename TYPE>
  class Ptensors0b:  public cnine::Ltensor<TYPE>, public cnine::diff_class<Ptensors0b<TYPE> >{
  public:

    using cnine::diff_class<Ptensors0b<TYPE> >::grad;

    typedef cnine::Ltensor<TYPE> BASE;
    using BASE::get_dev;


    AtomsPack0 atoms;


    ~Ptensors0b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0b(const AtomsPack& _atoms, const int nc):
      BASE(cnine::Gdims(_atoms.size(),nc)),
      atoms(_atoms){}


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
      BASE::reset(BASE({atoms.size(),v.nc},v.fcode,v.dev));
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


  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors0b(const Ptensors0& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    #ifdef _WITH_ATEN
    Ptensors0b(const at::Tensor& T):
      BASE(T){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0b(const Ptensors0b& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms.size();
    }

    int nchannels() const{
      return BASE::dim(1);
    }

    int offset(const int i) const{
      return i;
    }

    int nrows(const int i) const{
      return 1;
    }

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    BASE tensor_of(const int i) const{
      return BASE::row(offset(i));
    }

    Ptensor0 operator()(const int i) const{
      return Ptensor0(cnine::RtensorA(tensor_of(i).view1()),atoms_of(i));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    static Ptensors0b<TYPE> gather(const Ptensors0b<TYPE>& x, const AtomsPack& a){
      Ptensors0b<TYPE> R(a,x.nchannels());
      R.gather(x);
      return R;
    }

    static Ptensors0b<TYPE> gather(const Ptensors1b<TYPE>& x, const AtomsPack& a){
      Ptensors0b<TYPE> R(a,x.nchannels());
      R.gather(x);
      return R;
    }

    static Ptensors0b<TYPE> gather(const Ptensors2b<TYPE>& x, const AtomsPack& a){
      Ptensors0b<TYPE> R(a,2*x.nchannels());
      R.gather(x);
      return R;
    }

    template<typename SOURCE>
    void gather(const SOURCE& x){
      (atoms.overlaps_mmap(x.atoms))(*this,x);
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

}


#endif 


    //Ptensors0b(){}

    //Ptensors0b(const int _nc, const int _dev=0):
    //Ptensors(1,_nc,_dev){}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ptensors0b(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
    //Ptensors(_atoms, cnine::Gdims({_nc}), dummy, _dev){
    //if(atoms.constk()>0) constk=atoms.constk();
    //}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ptensors0b(const cnine::Tensor<int>& M, const int _nc, const FILLTYPE& dummy, const int _dev=0):
    //Ptensors(AtomsPack(M), cnine::Gdims({_nc}), dummy, _dev){
    //if(atoms.constk()>0) constk=atoms.constk();
    //}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ptensors0b(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
    //Ptensors(AtomsPack(_n), cnine::Gdims({_nc}), dummy, _dev){
    //constk=1;
    //}


