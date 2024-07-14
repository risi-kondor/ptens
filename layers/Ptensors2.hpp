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

#ifndef _ptens_Ptensors2
#define _ptens_Ptensors2

#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "Ptensor2.hpp"
#include "Ptensors.hpp"
#include "AtomsPackTag.hpp"
#include "Ptensor2view.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors0;
  template<typename TYPE> class Ptensors1;


  template<typename TYPE>
  class Ptensors2: public Ptensors<TYPE>, public cnine::diff_class<Ptensors2<TYPE> >{
  public:

    friend class Ptensors0<TYPE>;
    friend class Ptensors1<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    using cnine::diff_class<Ptensors2<TYPE> >::grad;
    using BASE::get_dev;
    using TENSOR::dev;
    using TENSOR::get_arr;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::strides;
    using TENSOR::cols;
    using TENSOR::add;

    using BASE::nc;
    using BASE::atoms;
    using BASE::size;
    using BASE::atoms_of;
    using BASE::get_nc;

    //AtomsPack atoms;
    //shared_ptr<PtensorsJig2<int> > jig;
    AtomsPackTag2 tag;


    ~Ptensors2(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //Ptensors2(const TENSOR& M):
    //BASE(M.copy()){} // for diff_class

    Ptensors2(const AtomsPack& _atoms, const TENSOR& M):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors2(const AtomsPack& _atoms, const cnine::TensorView<TYPE>& M):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors2(const TENSOR& M, const AtomsPack& _atoms):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors2(const TENSOR& M, const AtomsPackTag2& _tag):
      BASE(_tag.obj->atoms.lock(),M),
      tag(_tag){}

    Ptensors2(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(_atoms,cnine::Gdims(_atoms.nrows2(),nc),0,_dev),
      tag(_atoms){}

    Ptensors2(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(_atoms,cnine::Gdims(_atoms.nrows2(),nc),fcode,_dev),
      tag(_atoms){}


    static Ptensors2 cat(const vector<Ptensors2>& list){
      vector<AtomsPack> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      if(ptens_global::cache_atomspack_cats) 
	return Ptensors2(cnine::Ltensor<TYPE>::stack(0,list),ptens_global::atomspack_cat_cache(v));
      return Ptensors2(cnine::Ltensor<TYPE>::stack(0,list),AtomsPack::cat(v));
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors2(const AtomsPack& _atoms, const Args&... args):
      BASE(_atoms),
      tag(_atoms){
      vparams v;
      unroller(v,args...);
      nc=v.nc;
      BASE::reset(BASE({atoms.nrows2(),v.nc},v.fcode,v.dev));
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


    Ptensors2 copy() const{
      return Ptensors2(TENSOR::copy(),atoms);
    }

    Ptensors2 copy(const int _dev) const{
      return Ptensors2(TENSOR::copy(_dev),atoms);
    }

    Ptensors2 zeros_like() const{
      return Ptensors2(TENSOR::zeros_like(),atoms);
    }

    Ptensors2 gaussian_like() const{
      return Ptensors2(BASE::gaussian_like(),atoms);
    }

    static Ptensors2 zeros_like(const Ptensors2& x){
      return Ptensors2(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors2 gaussian_like(const Ptensors2& x){
      return Ptensors2(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors2* new_zeros_like(const Ptensors2& x){
      return new Ptensors2(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    //Ptensors2(const Ptensors2& x):
    //BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
    //atoms(x.atoms){
    //BASE::view2().set(x.view_as_matrix().view2());
    //}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors2(const Ptensors2& x, const int _dev):
      BASE(x.atoms,x.copy(_dev)), 
      //atoms(x.atoms),
      tag(x.tag){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors2& get_grad(){
      return cnine::diff_class<Ptensors2<TYPE> >::get_grad();
    }

    const Ptensors2& get_grad() const{
      return cnine::diff_class<Ptensors2<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 2;
    }

    //int size() const{
    //return atoms.size();
    //}

    //int get_nc() const{
    //return BASE::dim(1);
    //}

    //int nchannels() const{
    //return BASE::dim(1);
    //}

    int size_of(const int i) const{
      return atoms.size_of(i);
    }

    //AtomsPack get_atoms() const{
    //return atoms;
    //}

    int offset(const int i) const{
      return atoms.row_offset2(i);
    }

    int offset1(const int i) const{
      return atoms.row_offset1(i);
    }

    //Atoms atoms_of(const int i) const{
    //return atoms(i);
    //}
    
    TENSOR tensor_of(const int i) const{
      int k=size_of(i);
      return TENSOR::rows(offset(i),k*k).reshape({k,k,get_nc()});
    }

    Ptensor2<TYPE> operator()(const int i) const{
      return Ptensor2(tensor_of(i).view3(),atoms_of(i));
    }

    Ptensor2view<TYPE> view_of(const int i, const vector<int>& ix) const{
      int nc=get_nc();
	return Ptensor2view<TYPE>(const_cast<TYPE*>(get_arr())+offset(i)*nc,nc,size_of(i)*nc,nc,1,ix,get_dev());
    }

    Ptensor2view<TYPE> view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      int nc=get_nc();
      return Ptensor2view<TYPE>(const_cast<TYPE*>(get_arr())+offset(i)*nc+offs,n,size_of(i)*nc,nc,1,ix,get_dev());
    }

    Rtensor3_view view3_of(const int i) const{
      int n=size_of(i);
      return Rtensor3_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0],n,n,get_nc(),strides[0]*n,strides[0],strides[1],dev);
    }

    Rtensor3_view view3_of(const int i, const int offs, const int m) const{
      int n=size_of(i);
      return Rtensor3_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0]+offs*strides[1],
	n,n,m,strides[0]*n,strides[0],strides[1],dev);
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static Ptensors2<float> linmaps(const SOURCE& x){
      Ptensors2<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const Ptensors1<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      broadcast1(x,2*nc);
    }

    void add_linmaps(const Ptensors2<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      broadcast1(x.reduce1(),4*nc);
      broadcast2(x,13*nc);
    }

    void add_linmaps_back(const Ptensors0<TYPE>& r){
      broadcast0_shrink(r);
    }

    void add_linmaps_back(const Ptensors1<TYPE>& r){
      int nc=get_nc();
      broadcast0_shrink(r.reduce0(0,2*nc));
      broadcast1_shrink(r.cols(2*nc,3*nc));
    }

    void add_linmaps_back(const Ptensors2<TYPE>& r){
      int nc=get_nc();
      broadcast0_shrink(r.reduce0_shrink(0,nc));
      broadcast1_shrink(r.reduce1_shrink(4*nc,3*nc));
      add(r.reduce2_shrink(13*nc,nc));
    }


  public: // ---- Message passing ---------------------------------------------------------------------------


    template<typename SOURCE>
    static Ptensors2<TYPE> gather(const AtomsPack& a, const SOURCE& x){
      int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
      Ptensors2<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x){
      add_gather(x,ptens_global::overlaps_cache(atoms,x.atoms));
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      add_gather_back(x,ptens_global::overlaps_cache(x.atoms,atoms));
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const TensorLevelMap& map){
      if(ptens_global::row_level_operations){
	rmap(x,map)(*this,x);
      }else{
	int nc=x.get_nc();
	if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	if constexpr(std::is_same<SOURCE,Ptensors1<TYPE> >::value){
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),2*nc);
	}
	if constexpr(std::is_same<SOURCE,Ptensors2<TYPE> >::value){
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),4*nc);
	  broadcast2(x.reduce2(map.atoms(),map.in()),map.out(),13*nc);
	}
      }
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const TensorLevelMap& map){
      if(ptens_global::row_level_operations){
	x.rmap(*this,map).inv()(*this,x);
      }else{
	int nc=get_nc();
	if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value)
	  broadcast0_shrink(x.reduce0(map.atoms(),map.out()),map.in());
	if constexpr(std::is_same<OUTPUT,Ptensors1<TYPE> >::value){
	  broadcast0_shrink(x.reduce0(map.atoms(),map.out(),0,2*nc),map.in());
	  broadcast1_shrink(x.reduce1(map.atoms(),map.out(),2*nc,3*nc),map.in());
	}
	if constexpr(std::is_same<OUTPUT,Ptensors2<TYPE> >::value){
	  broadcast0_shrink(x.reduce0_shrink(map.atoms(),map.out(),0,2*nc),map.in());
	  broadcast1_shrink(x.reduce1_shrink(map.atoms(),map.out(),4*nc,3*nc),map.in());
	  broadcast2(x.reduce2_shrink(map.atoms(),map.out(),13*nc,nc),map.in());
	}
      }
    }


  private:

    template<typename SOURCE>
    RowLevelMap& rmap(const SOURCE& x, const TensorLevelMap& tmap) const{
      return *ptens_global::rmap_cache(tag,x.tag,tmap.obj);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      TimedFn T("Ptensors2","reduce0",*this);
      int N=size();
      int nc=get_nc();
      int dev=get_dev();
      PTENS_CPUONLY();
      
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BASE R({N,2*nc},0,dev);
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

    BASE reduce0_shrink(const int offs, const int nc) const{
      TimedFn T("Ptensors2","reduce0_shrink",*this);
      int N=size();
      int dev=get_dev();
      PTENS_CPUONLY();
      
      cnine::using_vram_manager vv(ptens_global::vram_manager);
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


    BASE reduce1() const{
      TimedFn T("Ptensors2","reduce1",*this);
      int N=size();
      int nc=get_nc();
      int dev=get_dev();
      PTENS_CPUONLY();

      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BASE R({atoms.nrows1(),3*nc},0,dev);
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

    
    BASE reduce1_shrink(const int offs, const int nc) const{
      TimedFn T("Ptensors2","reduce1_shrink",*this);
      int N=size();
      int dev=get_dev();
      PTENS_CPUONLY();

      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BASE R({atoms.nrows1(),nc},0,dev);
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


    BASE reduce2_shrink(const int offs, const int nc) const{
      TimedFn T("Ptensors2","reduce2_shrink",*this);
      int N=size();
      int dev=get_dev();
      PTENS_CPUONLY();
      
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BASE R({dim(0),nc},0,dev);
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset(i);
	  int n=size_of(i);
	  r.block(roffs,0,n*n,nc)+=view3_of(i,offs,nc).fuse01();
	  r.block(roffs,0,n*n,nc)+=view3_of(i,offs+nc,nc).transp01().fuse01();
	}
      }
      return R;
    }


  public: // ---- Cumulative Reductions ----------------------------------------------------------------------


    void add_reduce0_to(const BASE& R) const{
      TimedFn T("Ptensors2","reduce0",*this);
      PTENS_ASSRT(R.ndims()==2);
      PTENS_ASSRT(R.dim(0)==size());
      PTENS_ASSRT(R.dim(1)==2*nc);
      PTENS_CPUONLY();
      int N=size();
      int dev=get_dev();
      Rtensor2_view r=R.view2();
      if(dev==0){
	Rtensor2_view r0=R.block(0,0,N,nc);
	Rtensor2_view r1=R.block(0,nc,N,nc);
	for(int i=0; i<N; i++){
	  view3_of(i).sum01_into(r0.slice0(i));
	  view3_of(i).diag01().sum0_into(r1.slice0(i));
	}
      }
    }


    void add_reduce0_shrink_to(const BASE& R, const int offs) const{
      TimedFn T("Ptensors2","reduce0_shrink",*this);
      PTENS_ASSRT(R.ndims()==2);
      PTENS_ASSRT(R.dim(0)==size());
      PTENS_CPUONLY();
      int N=size();
      int nc=R.dim(1);
      int dev=get_dev();
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  view3_of(i,offs,nc).sum01_into(r.slice0(i));
	  view3_of(i,offs+nc,nc).diag01().sum0_into(r.slice0(i));
	}
      }
    }


    void add_reduce1_to(const BASE& R) const{
      TimedFn T("Ptensors2","reduce1",*this);
      PTENS_ASSRT(R.ndims()==2);
      PTENS_ASSRT(R.dim(0)==atoms.nrows1());
      PTENS_ASSRT(R.dim(1)==3*nc);
      PTENS_CPUONLY();
      int N=size();
      int dev=get_dev();
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
    }

    
    void add_reduce1_shrink_to(const BASE& R, const int offs) const{
      TimedFn T("Ptensors2","reduce1_shrink",*this);
      PTENS_ASSRT(R.ndims()==2);
      PTENS_ASSRT(R.dim(0)==atoms.nrows1());
      PTENS_CPUONLY();
      int N=size();
      int nc=R.dim(1);
      int dev=get_dev();
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
    }


    void add_reduce2_shrink_to(const BASE& R, const int offs) const{
      TimedFn T("Ptensors2","reduce2_shrink",*this);
      PTENS_ASSRT(R.ndims()==2);
      PTENS_ASSRT(R.dim(0)==dim(0));
      PTENS_CPUONLY();
      int N=size();
      int nc=R.dim(1);
      int dev=get_dev();
      Rtensor2_view r=R.view2();
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset(i);
	  int n=size_of(i);
	  r.block(roffs,0,n*n,nc)+=view3_of(i,offs,nc).fuse01();
	  r.block(roffs,0,n*n,nc)+=view3_of(i,offs+nc,nc).transp01().fuse01();
	}
      }
    }


  public: // ---- Indexed reductions -------------------------------------------------------------------------


    Ptensors0<TYPE> reduce0(const AtomsPack& _atoms, const AindexPack& list) const{
      TimedFn T("Ptensors2","reduce0",*this,list,(list.count2+list.count1)*get_nc());
      int nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors0<TYPE> R(_atoms,2*nc,0,dev);
      add_reduce0_to(R,list);
      return R;
    }

    Ptensors0<TYPE> add_reduce0_to(const Ptensors0<TYPE>& R, const AindexPack& list) const{
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).sum01_into(R.view_of(i).block(0,nc));
	  view_of(list.tens(i),list.ix(i)).diag01().sum0_into(R.view_of(i).block(nc,nc));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_reduce0_cu(R,*this,list,0,nc,stream)));
    }


    Ptensors0<TYPE> reduce0_shrink(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
      TimedFn T("Ptensors2","reduce0",*this,list,(list.count2+list.count1)*nc);
      if(nc==0) nc=get_nc()/2;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors0<TYPE> R(_atoms,nc,0,dev);
      add_reduce0_shrink_to(R,list,offs);
      return R;
    }

    void add_reduce0_shrink_to(const Ptensors0<TYPE>& R, const AindexPack& list, const int offs=0) const{
      int nc=R.get_nc();
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,nc).sum01_into(R.view_of(i));
	  view_of(list.tens(i),list.ix(i),offs+nc,nc).diag01().sum0_into(R.view_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_reduce0_cu(R,*this,list,0,nc,stream)));
    }


    Ptensors1<TYPE> reduce1(const AtomsPack& _atoms, const AindexPack& list) const{
      TimedFn T("Ptensors2","reduce1",*this,list,(list.count1+2*list.count2)*get_nc());
      int nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors1<TYPE> R(_atoms,3*nc,0,dev);
      add_reduce1_to(R,list);
      return R;
    }

    void add_reduce1_to(const Ptensors1<TYPE>& R, const AindexPack& list) const{
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).sum0_into(R.view_of(i).block(0,0,-1,nc));
	  view_of(list.tens(i),list.ix(i)).sum1_into(R.view_of(i).block(0,nc,-1,nc));
	  R.view_of(i).block(0,2*nc,-1,nc)+=view_of(list.tens(i),list.ix(i)).diag01(); // is this a problem?
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_reduce1_cu(R,*this,list,0,nc,stream)));
    }


    Ptensors1<TYPE> reduce1_shrink(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
      TimedFn T("Ptensors2","reduce1",*this,list,(list.count1+2*list.count2)*nc);
      if(nc==0) nc=get_nc()/3;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors1<TYPE> R(_atoms,nc,0,dev);
      add_reduce1_shrink_to(R,list,offs,nc);
      return R;
    }

    void add_reduce1_shrink_to(const Ptensors1<TYPE>& R, const AindexPack& list, const int offs=0, int nc=0) const{
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,nc).sum0_into(R.view_of(i));
	  view_of(list.tens(i),list.ix(i),offs+nc,nc).sum1_into(R.view_of(i));
	  R.view_of(i)+=view_of(list.tens(i),list.ix(i),offs+2*nc,nc).diag01(); // is this a problem?
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_reduce1_cu(R,*this,list,0,nc,stream)));
    }


    Ptensors2<TYPE> reduce2(const AtomsPack& _atoms, const AindexPack& list) const{
      TimedFn T("Ptensors2","reduce2",*this,list,(2*list.count2)*get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors2<TYPE> R(_atoms,get_nc(),0,dev);
      add_reduce2_to(R,list);
      return R;
    }

    void add_reduce2_to(const Ptensors2<TYPE>& R, const AindexPack& list) const{
      if(dev==0){
	int nc=get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view3_of(i,0,nc)+=view_of(list.tens(i),list.ix(i));
	  R.view3_of(i,nc,nc)+=view_of(list.tens(i),list.ix(i)).transp();
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_reduce2B_cu(R,*this,list,offs,n,stream)));
    }


    Ptensors2<TYPE> reduce2_shrink(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
      TimedFn T("Ptensors2","reduce2",*this,list,(2*list.count2)*nc);
      if(nc==0) nc=get_nc()/2;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors2<TYPE> R(_atoms,nc,0,dev);
      add_reduce2_shrink_to(R,list,offs,nc);
      return R;
    }

    void add_reduce2_shrink_to(const Ptensors2<TYPE>& R, const AindexPack& list, const int offs=0, int nc=0) const{
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs,nc);
	  R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs+nc,nc).transp();
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_reduce2B_cu(R,*this,list,offs,n,stream)));
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      TimedFn T("Ptensors2","broadcast0",*this);
      int N=size();
      int dev=get_dev();
      int nc=X.dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      Rtensor2_view x=X.view2();
      
      if(dev==0){
	for(int i=0; i<N; i++){
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=repeat0(repeat0(x.slice0(i),n),n);
	  view3_of(i,offs+nc,nc).diag01()+=repeat0(x.slice0(i),n);
	}
      }
    }

    void broadcast0_shrink(const BASE& X){
      TimedFn T("Ptensors2","broadcast0_shrink",*this);
      int N=size();
      int dev=get_dev();
      int nc=dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      PTENS_ASSRT(X.dim(1)==2*nc);
      Rtensor2_view x=X.view2();
      Rtensor2_view x0=x.block(0,0,N,nc);
      Rtensor2_view x1=x.block(0,nc,N,nc);
      
      if(dev==0){
	for(int i=0; i<N; i++){
	  int n=size_of(i);
	  view3_of(i)+=repeat0(repeat0(x0.slice0(i),n),n);
	  view3_of(i).diag01()+=repeat0(x1.slice0(i),n);
	}
      }
    }
    

    void broadcast1(const BASE& X, const int offs=0){
      TimedFn T("Ptensors2","broadcast1",*this);
      int N=size();
      int dev=get_dev();
      int nc=X.dim(1);
      Rtensor2_view x=X.view2();

      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=repeat0(x.block(roffs,0,n,nc),n);
	  view3_of(i,offs+nc,nc)+=repeat1(x.block(roffs,0,n,nc),n);
	  view3_of(i,offs+2*nc,nc).diag01()+=x.block(roffs,0,n,nc);
	}
      }
    }


    void broadcast1_shrink(const BASE& X){
      TimedFn T("Ptensors2","broadcast1_shrink",*this);
      int N=size();
      int dev=get_dev();
      int nc=dim(1);
      PTENS_ASSRT(X.dim(1)==3*nc);
      Rtensor2_view x=X.view2();
      Rtensor2_view x0=x.block(0,0,X.dim(0),nc);
      Rtensor2_view x1=x.block(0,nc,X.dim(0),nc);
      Rtensor2_view x2=x.block(0,2*nc,X.dim(0),nc);
      

      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset1(i);
	  int n=size_of(i);
	  view3_of(i)+=repeat0(x.block(roffs,0,n,nc),n);
	  view3_of(i)+=repeat1(x.block(roffs,nc,n,nc),n);
	  view3_of(i).diag01()+=x.block(roffs,2*nc,n,nc);
	}
      }
    }


    void broadcast2(const BASE& X, const int offs=0){
      TimedFn T("Ptensors2","broadcast2",*this);
      int N=size();
      int dev=get_dev();
      int nc=X.dim(1);
      Rtensor2_view x=X.view2();
      
      if(dev==0){
	for(int i=0; i<N; i++){
	  int roffs=offset(i);
	  int n=size_of(i);
	  view3_of(i,offs,nc)+=split0(x.block(roffs,0,n*n,nc),n,n);
	  view3_of(i,offs+nc,nc)+=split0(x.block(roffs,0,n*n,nc),n,n).transp01();
	}
      }
    }


  public: // ---- Idexed broadcasting -------------------------------------------------------------------------------


    void broadcast0(const Ptensors0<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors2","broadcast0",*this,x,list,(list.count1+list.count2)*x.get_nc());
      if(dev==0){
	const int n=x.get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue; // probably redundant
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(repeat0(x.view_of(i),list.nix(i)),list.nix(i));
	  view_of(list.tens(i),list.ix(i),offs+n,n).diag01()+=repeat0(x.view_of(i),list.nix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_broadcast0_cu(*this,x,list,offs,stream)));
    }

    void broadcast1(const Ptensors1<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors2","broadcast1",*this,x,list,(list.count1+2*list.count2)*x.get_nc());
      if(dev==0){
	const int n=x.get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view_of(i),list.nix(i));
	  view_of(list.tens(i),list.ix(i),offs+n,n)+=repeat1(x.view_of(i),list.nix(i));
	  view_of(list.tens(i),list.ix(i),offs+2*n,n).diag01()+=x.view_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_broadcast1_cu(*this,x,list,offs,stream)));
    }

    void broadcast2(const Ptensors2<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors2","broadcast2",*this,x,list,(2*list.count2)*x.get_nc());
      if(dev==0){
	const int n=x.get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n)+=x.view3_of(i);
	  view_of(list.tens(i),list.ix(i),offs+n,n)+=x.view3_of(i).transp01();
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_broadcast2_cu(*this,x,list,offs,stream)));
    }


    void broadcast0_shrink(const Ptensors0<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors2","broadcast0_shrink",*this,x,list,(list.count1+list.count2)*get_nc());
      if(dev==0){
	const int n=get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue; // probably redundant
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(repeat0(x.view_of(i,0,n),list.nix(i)),list.nix(i));
	  view_of(list.tens(i),list.ix(i),offs,n).diag01()+=repeat0(x.view_of(i,n,n),list.nix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_broadcast0_cu(*this,x,list,offs,stream)));
    }

    void broadcast1_shrink(const Ptensors1<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors2","broadcast1_shrink",*this,x,list,(list.count1+2*list.count2)*x.get_nc());
      if(dev==0){
	const int n=get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view_of(i,0,n),list.nix(i));
	  view_of(list.tens(i),list.ix(i),offs,n)+=repeat1(x.view_of(i,n,n),list.nix(i));
	  view_of(list.tens(i),list.ix(i),offs,n).diag01()+=x.view_of(i,2*n,n);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_broadcast1_cu(*this,x,list,offs,stream)));
    }

    /*
    void broadcast2_shrink(const Ptensors2<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors2","broadcast2_shrink",*this,x,list,(2*list.count2)*x.get_nc());
      if(dev==0){
	const int n=get_nc();
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,n)+=x.view3_of(i);
	  view_of(list.tens(i),list.ix(i),offs+n,n)+=x.view3_of(i).transp01();
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors2_broadcast2_cu(*this,x,list,offs,stream)));
    }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors2";
    }

    string repr() const{
      return "<Ptensors2[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors2 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors2& x){
      stream<<x.str(); return stream;}



  };


  template<typename SOURCE>
  inline Ptensors2<float> linmaps2(const SOURCE& x){
    Ptensors2<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  Ptensors2<float> gather2(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
    Ptensors2<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }

}


#endif 

    /*
    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){
      auto overlaps=ptens_global::overlaps_cache(x.atoms,atoms);
      if(ptens_global::row_level_operations){
	x.rmap(*this,overlaps).inv()(get_grad(),x.get_grad());
      }else{
      }
    }
    */
