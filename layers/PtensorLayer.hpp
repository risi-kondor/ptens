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
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"


namespace ptens{


  template<typename TYPE>
  class PtensorLayer: public cnine::Ltensor<TYPE>, public cnine::diff_class<PtensorLayer<TYPE> >{
  public:

    using cnine::diff_class<PtensorLayer<TYPE> >::grad;
    using cnine::diff_class<PtensorLayer<TYPE> >::get_grad;

    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    typedef cnine::Ltensor<TYPE> BASE;
    using BASE::get_dev;
    using BASE::dim;
    using BASE::move_to_device;
    //using BASE::add;
    //using BASE::mprod;
    using BASE::inp;
    using BASE::diff2;
    //using BASE::block;

    using BASE::dev;
    using BASE::strides;
    using BASE::get_arr;

#ifdef _WITH_CUDA
    using BASE::torch;
#endif 

    AtomsPackN atoms;
    int nc=0;


    ~PtensorLayer(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    PtensorLayer(const int k, const BASE& M):
      BASE(M.copy()),
      atoms(k,AtomsPack(M.dim(0))),
      nc(M.dim(1)){}

    PtensorLayer(const int k, const AtomsPack& _atoms, const BASE& M):
      BASE(M.copy()),
      atoms(k,_atoms),
      nc(M.dim(1)){}

    PtensorLayer(const int k, const AtomsPack& _atoms, const int _nc, const int _dev=0):
      atoms(k,_atoms), nc(_nc){
      BASE::reset(BASE({atoms.nrows(),nc},0,_dev));
    }

    PtensorLayer(const int k, const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev):
      atoms(k,_atoms), nc(_nc){
      BASE::reset(BASE({atoms.nrows(),nc},fcode,_dev));
    }

    PtensorLayer(const BASE& x, const AtomsPackN& _atoms):
      BASE(x), atoms(_atoms), nc(x.dim(1)){}

    PtensorLayer(const AtomsPackN& _atoms, const int _nc, const int _dev=0):
      BASE({atoms.nrows(),_nc},0,_dev),
      atoms(_atoms),
      nc(_nc){}


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
      atoms(0,x.atoms), nc(x.nc){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    PtensorLayer(const Ptensors1& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(1,x.atoms), nc(x.nc){
      BASE::view2().set(x.view_as_matrix().view2());
    }

    PtensorLayer(const Ptensors2& x):
      BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
      atoms(2,x.atoms), nc(x.nc){
      BASE::view2().set(x.view_as_matrix().view2());
    }

#ifdef _WITH_ATEN // needed for grad
    PtensorLayer(const at::Tensor& T):
      BASE(T){}
#endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    PtensorLayer(const PtensorLayer& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms), nc(x.nc){}


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

    int size1() const{
      CNINE_UNIMPL();
      return 0;
    }

    int get_nc() const{
      return nc;
      //      return BASE::dim(1);
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

    int offset1(const int i) const{
      return atoms.offset1(i);
    }

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    BASE tensor_of(const int i) const{
      int k=getk();
      if(k==1) return BASE::slices(0,offset(i),size_of(i));
      if(k==2) {int n=size_of(i); return BASE::slices(0,offset(i),n*n).reshape({n,n,get_nc()});}
      return BASE::slice(0,offset(i));
    }

    BASE operator()(const int i) const{
      return tensor_of(i);
    }

    Rtensor1_view view1_of(const int i) const{
      return Rtensor1_view(get_arr(),nc,strides[1],dev);
    }

    Rtensor1_view view1_of(const int i, const int offs, const int m) const{
      return Rtensor1_view(get_arr()+offs*strides[1],
	m,strides[1],dev);
    }

    Rtensor2_view view2_of(const int i) const{
      return Rtensor2_view(get_arr()+offset(i)*strides[0],size_of(i),nc,strides[0],strides[1],dev);
    }

    Rtensor2_view view2_of(const int i, const int offs, const int m) const{
      return Rtensor2_view(get_arr()+offset(i)*strides[0]+offs*strides[1],
	size_of(i),m,strides[0],strides[1],dev);
    }

    Rtensor3_view view3_of(const int i) const{
      int n=size_of(i);
      return Rtensor3_view(get_arr()+offset(i)*strides[0],n,n,nc,strides[0]*n,strides[0],strides[1],dev);
    }

    Rtensor3_view view3_of(const int i, const int offs, const int m) const{
      int n=size_of(i);
      return Rtensor3_view(get_arr()+offset(i)*strides[0]+offs*strides[1],
	n,n,m,strides[0]*n,strides[0],strides[1],dev);
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE>
    static PtensorLayer<TYPE> gather(const int k, const AtomsPack& a, const SOURCE& x){
      int m=vector<int>({1,1,2,5,15})[x.getk()+k];
      PtensorLayer<TYPE> R(k,a,m*x.get_nc(),x.get_dev());
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


    static PtensorLayer linmaps(const int k, const PtensorLayer& x){
      int xk=x.getk();
      int nc=x.get_nc();
      int nc_out=vector<int>({1,1,2,5,15})[k+xk]*nc;
      PtensorLayer<TYPE> r(k,x.atoms,nc_out,x.get_dev());
      if(k==0){
	if(xk==0) r+=x;
	if(xk==1) r+=x.reduce0();
	if(xk==2) r+=x.reduce0();
      }
      if(k==1){
	if(xk==0) 
	  r.broadcast0(x);
	if(xk==1){
	  r.broadcast0(x.reduce0());
	  r.cols(nc,nc)+=x;}
	if(xk==2){
	  r.broadcast0(x.reduce0());
	  r.cols(2*nc,3*nc)+=x.reduce1();}
      }
      if(k==2){
	if(xk==0) 
	  r.broadcast0(x);
	if(xk==1){
	  r.broadcast0(x.reduce0());
	  r.broadcast1(x,2*nc);}
	if(xk==2){
	  r.broadcast0(x.reduce0());
	  r.broadcast1(x.reduce1(),4*nc);
	  r.broadcast2(x,13*nc);
	}
      }
      return r;
    }


    void linmaps_back(const PtensorLayer& r){
      int k=r.getk();
      int xk=getk();
      int nc=get_nc();
      int nc_out=vector<int>({1,1,2,5,15})[k+xk]*nc;
      PTENS_ASSRT(r.dim(1)==nc_out);
      if(k==0){
	if(xk==0) add(r);
	if(xk==1) broadcast0(r);
	if(xk==2) broadcast0_shrink(r);
      }
      if(k==1){
	if(xk==0) 
	  add(r.reduce0());
	if(xk==1){
	  broadcast0(r.reduce0(0,nc));
	  add(r.cols(nc,nc));}
	if(xk==2){
	  broadcast0_shrink(r.reduce0(0,2*nc));
	  broadcast0_shrink(r.cols(2*nc,3*nc));}
      }
      if(k==2){
	if(xk==0) 
	  add(r.reduce0_shrink(0,nc));
	if(xk==1){
	  broadcast0(r.reduce0_shrink(0,nc));
	  add(r.reduce1_shrink(2*nc,nc));}
	if(xk==2){
	  broadcast0_shrink(r.reduce0_shrink(0,2*nc));
	  broadcast1_shrink(r.reduce1_shrink(4*nc,3*nc));
	  add(r.reduce2_shrink(13*nc,nc));}
      }
      return r;
    }

    
  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      TimedFn T("PtensorLayer"+to_string(getk()),"reduce0",*this);
      int N=size();
      int dev=get_dev();
      
      if(getk()==0) return *this;
      
      if(getk()==1){
	BASE R({N,nc},0,dev);
	Rtensor2_view r=R.view2();
	if(dev==0){
	  for(int i=0; i<N; i++)
	    view2_of(i).sum0_into(r.slice0(i));
	}
	return R;
      }
      
      if(getk()==2){
	BASE R({N,3*nc},0,dev);
	Rtensor2_view r=R.view2();
	if(dev==0){
	Rtensor2_view r0=R.block(0,0,N,nc);
	Rtensor2_view r1=R.block(0,nc,N,nc);
	for(int i=0; i<N; i++){
	  view3_of(i).sum01_into(r0.slice0(i));
	  view3_of(i).diag01().sum0_into(r1.slice0(i));
	}
      }
      return R;
    }

    CNINE_UNIMPL();
    return BASE();
    }


    BASE reduce0_shrink(const int offs, const int nc) const{
      TimedFn T("PtensorLayer"+to_string(getk()),"reduce0_shrink",*this);
      int N=size();
      int dev=get_dev();
      
      if(getk()==2){
	BASE R({N,nc},0,dev);
	Rtensor2_view r=R.view2();
	if(dev==0){
	  for(int i=0; i<N; i++){
	    view3_of(i,offs,nc).sum01_into(r.slice0(i));
	    view3_of(i,offs+nc,nc).diag01().sum0_into(r.slice0(i));
	  }
	}
	return R;
      }

      CNINE_UNIMPL();
      return BASE();
    }


    BASE reduce1() const{
      TimedFn T("PtensorLayer"+to_string(getk()),"reduce1",*this);
      int N=size();
      int dev=get_dev();
      
      if(getk()==1) return *this;
      
      if(getk()==2){
	BASE R({size1(),3*nc},0,dev);
	Rtensor2_view r=R.view2();
	if(dev==0){
	  for(int i=0; i<N; i++){
	    int roffs=offset1(i);
	    int n=size_of(i);
	    view3_of(i).sum0_into(r.block(roffs,0,n,nc));
	    view3_of(i).sum1_into(r.block(roffs,nc,n,nc));
	    r.block(roffs,2*nc,n,nc)+=view3_of(i).diag01();
	  }
	}
	return R;
      }
      
      CNINE_UNIMPL();
      return BASE();
    }

    
    BASE reduce1_shrink(const int offs, const int nc) const{
      TimedFn T("PtensorLayer"+to_string(getk()),"reduce1_shrink",*this);
      int N=size();
      int dev=get_dev();

      if(getk()==2){
	BASE R({dim(0),nc},0,dev);
	Rtensor2_view r=R.view2();
	if(dev==0){
	  for(int i=0; i<N; i++){
	    int roffs=offset1(i);
	    int n=size_of(i);
	    view3_of(i,offs,nc).sum0_into(r.block(roffs,0,n,nc));
	    view3_of(i,offs+nc,nc).sum1_into(r.block(roffs,0,n,nc));
	    r.block(roffs,0,n,nc)+=view3_of(i,offs+2*nc,nc).diag01();
	  }
	}
	return R;
      }
      
      CNINE_UNIMPL();
      return BASE();
    }


    BASE reduce2_shrink(const int offs, const int nc) const{
      TimedFn T("PtensorLayer"+to_string(getk()),"reduce2_shrink",*this);
      int N=size();
      int dev=get_dev();
      
      if(getk()==2){
	BASE R({dim(0),nc},0,dev);
	Rtensor2_view r=R.view2();
	if(dev==0){
	  for(int i=0; i<N; i++){
	    int roffs=offset(i);
	    int n=size_of(i);
	    r.block(roffs,0,n*n,nc)+=view3_of(i,offs,nc);
	    r.block(roffs,0,n*n,nc)+=view3_of(i,offs+nc,nc).transp01();
	  }
	}
	return R;
      }
      
      CNINE_UNIMPL();
      return BASE();
    }



  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
    TimedFn T("PtensorLayer"+to_string(getk()),"broadcast0",*this);
    int N=size();
    int dev=get_dev();
    int nc=X.dim(1);
    PTENS_ASSRT(X.dim(0)==N);
    Rtensor2_view x=X.view2();

    if(getk()==0){
      BASE::view2().block(0,offs,N,nc)+=x;
      return;
    }

    if(getk()==1){
      if(dev==0){
	for(int i=0; i<N; i++)
	  view2_of(i,offs,nc)+=cnine::repeat0(x.slice0(i),size_of(i));
      }
      return; 
    }

    if(getk()==2){
      if(dev==0){
	for(int i=0; i<N; i++){
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=repeat0(repeat0(x.slice0(i),n),n);
	  view3_of(i,offs+nc,nc).diag01()+=repeat0(x.slice0(i),n);
	}
      }
      return; 
    }

    CNINE_UNIMPL();
  }

  void broadcast0_shrink(const BASE& X){
    TimedFn T("PtensorLayer"+to_string(getk()),"broadcast0_shrink",*this);
    int N=size();
    int dev=get_dev();
    int nc=dim(1);
    PTENS_ASSRT(X.dim(0)==N);
    PTENS_ASSRT(X.dim(1)==2*nc);
    Rtensor2_view x=X.view2();
    Rtensor2_view x0=x.block(0,0,N,nc);
    Rtensor2_view x1=x.block(0,nc,N,nc);

    if(getk()==2){
      if(dev==0){
	for(int i=0; i<N; i++){
	  int n=size_of(i);
	  view3_of(i)+=repeat0(repeat0(x0.slice0(i),n),n);
	  view3_of(i).diag01()+=repeat0(x1.slice0(i),n);
	}
      }
      return; 
    }

    CNINE_UNIMPL();
  }


  void broadcast1(const BASE& X, const int offs=0){
    TimedFn T("PtensorLayer"+to_string(getk()),"broadcast1",*this);
    int N=size();
    int dev=get_dev();
    int nc=X.dim(1);
    Rtensor2_view x=X.view2();

    if(getk()==1){
      BASE::view2().block(0,offs,dim(0),nc)+=x;
      return; 
    }

    if(getk()==2){
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=repeat0(x.block(roffs,0,n,nc),n);
	  view3_of(i,offs+nc,nc)+=repeat1(x.block(roffs,0,n,nc),n);
	  view3_of(i,offs+2*nc,nc).diag01()+=x.block(roffs,0,n,nc);
	}
      }
      return; 
    }

    CNINE_UNIMPL();
  }

  void broadcast1_shrink(const BASE& X){
    TimedFn T("PtensorLayer"+to_string(getk()),"broadcast1_shrink",*this);
    int N=size();
    int dev=get_dev();
    int nc=X.dim(1);
    PTENS_ASSRT(X.dim(1)==3*nc);
    Rtensor2_view x=X.view2();
    Rtensor2_view x0=x.block(0,0,X.dim(0),nc);
    Rtensor2_view x1=x.block(0,nc,X.dim(0),nc);
    Rtensor2_view x2=x.block(0,2*nc,X.dim(0),nc);


   if(getk()==2){
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i)+=repeat0(x.block(roffs,0,n,nc),n);
	  view3_of(i)+=repeat1(x.block(roffs,nc,n,nc),n);
	  view3_of(i).diag01()+=x.block(roffs,2*nc,n,nc);
	}
      }
      return; 
    }

    CNINE_UNIMPL();
  }


  void broadcast2(const BASE& X, const int offs=0){
    TimedFn T("PtensorLayer"+to_string(getk()),"broadcast2",*this);
    int N=size();
    int dev=get_dev();
    int nc=X.dim(1);
    Rtensor2_view x=X.view2();

    if(getk()==2){
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=x.block(roffs,0,n*n,nc);
	  // view3_of(i,offs+nc,nc)+=x.block(roffs,0,n*n,nc).transp01(); // todo!
	}
      }
      return; 
    }

    CNINE_UNIMPL();
  }


  public: // ---- Operations ---------------------------------------------------------------------------------


    static PtensorLayer cat(const vector<reference_wrapper<PtensorLayer> >& list){
      vector<shared_ptr<AtomsPackObjBase> > v; 
      for(auto p:list)
	v.push_back(p.get().atoms.obj);
      return PtensorLayer(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    }

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

    /*
    static PtensorLayer cat(const vector<PtensorLayer*>& list){
      vector<shared_ptr<AtomsPackObjBase> > v; 
      for(auto p:list)
	v.push_back(p->atoms->obj);
      return PtensorLayer(BASE::stack(0,list),AtomsPackObjBase::cat(v));
    }
    */

