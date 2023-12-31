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
    using BASE::torch;
    using BASE::get_dev;
    using BASE::dim;
    using BASE::move_to_device;
    //using BASE::add;
    //using BASE::mprod;
    using BASE::inp;
    using BASE::diff2;
    //using BASE::block;

    AtomsPackN atoms;


    ~PtensorLayer(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    PtensorLayer(const int k, const BASE& M):
      BASE(M.copy()),
      atoms(k,AtomsPack(M.dim(0))){}

    PtensorLayer(const int k, const AtomsPack& _atoms, const BASE& M):
      BASE(M.copy()),
      atoms(k,_atoms){}

    PtensorLayer(const int k, const AtomsPack& _atoms, const int nc, const int _dev=0):
      atoms(k,_atoms){
      BASE::reset(BASE({atoms.nrows(),nc},0,_dev));
    }

    PtensorLayer(const int k, const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      atoms(k,_atoms){
      BASE::reset(BASE({atoms.nrows(),nc},fcode,_dev));
    }

    PtensorLayer(const BASE& x, const AtomsPackN& _atoms):
      BASE(x), atoms(_atoms){}


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


    PtensorLayer copy() const{
      return PtensorLayer(BASE::copy(),atoms);
    }

    PtensorLayer copy(const int _dev) const{
      return PtensorLayer(BASE::copy(_dev),atoms);
    }

    PtensorLayer zeros_like() const{
      return PtensorLayer(BASE::zeros_like(),atoms);
    }

    PtensorLayer gaussian_like() const{
      return PtensorLayer(BASE::gaussian_like(),atoms);
    }

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

#ifdef _WITH_ATEN // needed for grad
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


  public: // ---- Operations ---------------------------------------------------------------------------------


    static PtensorLayer cat(const vector<reference_wrapper<PtensorLayer> >& list){
      vector<shared_ptr<AtomsPackObjBase> > v; 
      for(auto p:list)
	v.push_back(p.get().atoms.obj);
      return PtensorLayer(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    }

    /*
    static PtensorLayer cat(const vector<PtensorLayer*>& list){
      vector<shared_ptr<AtomsPackObjBase> > v; 
      for(auto p:list)
	v.push_back(p->atoms->obj);
      return PtensorLayer(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    }
    */

    PtensorLayer cat_channels(const PtensorLayer& y) const{
      PTENS_ASSRT(atoms==y.atoms);
      PTENS_ASSRT(dim(0)==y.dim(0));
      BASE R({dim(0),dim(1)+y.dim(1)},0,get_dev());
      R.block(0,0,dim(0),dim(1))+=*this;
      R.block(0,dim(1),dim(0),y.dim(1))+=y;
      return PtensorLayer(R,atoms);
    }

    PtensorLayer mult_channels(const cnine::Ltensor<TYPE>& s) const{
      return PtensorLayer(BASE::scale_columns(s),atoms);
    }
    
    PtensorLayer mprod(const cnine::Ltensor<TYPE>& M) const{
      return PtensorLayer(mult(*this,M),atoms);
    }
    
    PtensorLayer linear(const PtensorLayer& x, const cnine::Ltensor<TYPE>& w, const cnine::Ltensor<TYPE>& b) const{
      PtensorLayer R(x*w,x.atoms);
      R.add_broadcast(0,b);
      return R;
    }

    PtensorLayer ReLU(TYPE alpha) const{
      return PtensorLayer(BASE::ReLU(alpha),atoms);
    }

    void cat_channels_back0(const PtensorLayer& g){
      get_grad()+=g.get_grad().block(0,0,dim(0),dim(1));
    }

    void cat_channels_back1(const PtensorLayer& g){
      get_grad()+=g.get_grad().block(0,g.dim(1)-dim(1),dim(0),dim(1));
    }

    void add_mult_channels_back(const PtensorLayer& g, const BASE& s){
      get_grad().BASE::add_scale_columns(g.get_grad(),s);
    }

    void add_mprod_back0(const PtensorLayer& g, const BASE& M){
      get_grad().BASE::add_mprod(g.get_grad(),M.transp());
    }

    void add_linear_back0(const PtensorLayer& g, const BASE& M){
      get_grad().BASE::add_mprod(g.get_grad(),M.transp());
    }

    void add_ReLU_back(const PtensorLayer& g, const PtensorLayer& x, const float alpha){
      get_grad().BASE::add_ReLU_back(g.get_grad(),x,alpha);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorLayer";
    }

    string repr() const{
      return "<PtensorLayer[k="+to_string(getk())+",N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	PtensorLayer y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Ptensor"<<getk()<<"("<<atoms_of(i)<<"):"<<endl;
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

