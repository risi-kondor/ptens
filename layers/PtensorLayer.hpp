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

#ifndef _ptens_PtensorLayer
#define _ptens_PtensorLayer

#include "diff_class.hpp"
#include "AtomsPackN.hpp"
#include "Ptensors1.hpp"
#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensors0b.hpp"
#include "Ptensors1b.hpp"
#include "Ptensors2b.hpp"


namespace ptens{


  template<typename TYPE>
  class PtensorLayer: public cnine::Ltensor<TYPE>, public cnine::diff_class<PtensorLayer<TYPE> >{
  public:

    using cnine::diff_class<PtensorLayer<TYPE> >::grad;
    using cnine::diff_class<PtensorLayer<TYPE> >::get_grad;

    typedef cnine::Ltensor<TYPE> BASE;
    using BASE::get_dev;


    AtomsPackN atoms;


    ~PtensorLayer(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    PtensorLayer(const int k, const AtomsPack& _atoms, const int nc, const int _dev=0):
      atoms(k,_atoms){
      BASE::reset(BASE({atoms.nrows(),nc},0,_dev));
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    PtensorLayer(const int k, const AtomsPack& _atoms, const Args&... args):
      atoms(k,_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.nrows(),v.nc},v.fcode,v.dev));
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


    static PtensorLayer* new_zeros_like(const PtensorLayer& x){
      return new PtensorLayer(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    PtensorLayer(const Ptensors0& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(0,x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    PtensorLayer(const Ptensors1& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(1,x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    PtensorLayer(const Ptensors2& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(2,x.atoms){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    #ifdef _WITH_ATEN
    PtensorLayer(const at::Tensor& T):
      BASE(T){}
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    PtensorLayer(const PtensorLayer& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Access ------------------------------------------------------------------------------------


    int getk() const{
      return atoms.getk();
    }

    /*
    void switch_k(std::function<void()>& lambda0, std::function<void()>& lambda1, std::function<void()>& lambda2){
      int k=getk();
      if(k==0) lambda0();
      if(k==1) lambda1();
      if(k==2) lambda2();
    }
    */

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
      int k=getk();
      if(k==1) return BASE::slices(0,offset(i),size_of(i));
      if(k==2) {int n=size_of(i); return BASE::slices(0,offset(i),n*n).reshape({n,n,nchannels()});}
      return BASE::slice(0,offset(i));
    }

    BASE operator()(const int i) const{
      return tensor_of(i);
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE>
    static PtensorLayer<TYPE> gather(const int k, const AtomsPack& a, const SOURCE& x){
      int nc=vector<int>({1,1,2,5,15})[x.getk()+k];
      PtensorLayer<TYPE> R(k,a,nc*x.nchannels(),x.get_dev());
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
      return "PtensorLayer";
    }

    string repr() const{
      return "<PtensorLayer[k="<<getk()<<",N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	PtensorLayer y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Ptensor"<<getk()<<atoms_of(i)<<":"<<endl;
	oss<<tensor_of(i).to_string(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const PtensorLayer& x){
      stream<<x.str(); return stream;}



  };


}


#endif 

    /*
    static PtensorLayer<TYPE> gather(const int k, const AtomsPack& a, const Ptensors0b<TYPE>& x){
      int nc=vector<int>({1,1,2})[k];
      PtensorLayer<TYPE> R(k,a,nc*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static PtensorLayer<TYPE> gather(const int k, const AtomsPack& a, const Ptensors1b<TYPE>& x){
      int nc=vector<int>({1,2,5})[k];
      PtensorLayer<TYPE> R(k,a,nc*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }

    static PtensorLayer<TYPE> gather(const int k, const AtomsPack& a, const Ptensors2b<TYPE>& x){
      int nc=vector<int>({2,5,15})[k];
      PtensorLayer<TYPE> R(k,a,nc*x.nchannels(),x.get_dev());
      R.gather(x);
      return R;
    }
    */

