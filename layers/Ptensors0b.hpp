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
#include "Ptensorsb.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors1b;
  template<typename TYPE> class Ptensors2b;


  template<typename TYPE>
  class Ptensors0b: public Ptensorsb<TYPE, Ptensors0b<TYPE> >, public cnine::diff_class<Ptensors0b<TYPE> >{

  public:

    typedef Ptensorsb<TYPE, Ptensors0b<TYPE> > BASE;
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


    Ptensors0b(const TENSOR& M):
      BASE(M.copy()),
      atoms(AtomsPack0(M.dim(0))){}

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

#ifdef _WITH_ATEN
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
#endif 


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

    int offset(const int i) const{
      return i;
    }

    int nrows(const int i) const{
      return 1;
    }

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


    //Ptensors0b mprod(const TENSOR& y){
    //return Ptensors0b(BASE::mprod(y),atoms);
    //}

    //Ptensors0b scale_channels(const TENSOR& s){
    //return Ptensors0b(BASE::scale_channels(s),atoms);
    //}


  public: // ---- Message passing ----------------------------------------------------------------------------


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
      x.atoms.overlaps_mmap(atoms).inv()(get_grad(),x.get_grad());
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


