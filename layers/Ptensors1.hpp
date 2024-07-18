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
#include "Ptensors.hpp"
#include "AtomsPackTag.hpp"
#include "Ptensor1view.hpp"
#include "PtensorMap.hpp"
#include "PtensorMapFactory.hpp"


namespace ptens{


  template<typename TYPE>
  class Ptensors1: public Ptensors<TYPE>, public cnine::diff_class<Ptensors1<TYPE> >{
  public:

    friend class Ptensors0<TYPE>;
    friend class Ptensors2<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor1_view Rtensor1_view;
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

    using BASE::nc;
    using BASE::atoms;
    using BASE::size;
    using BASE::atoms_of;
    using BASE::get_nc;


    //AtomsPack atoms;
    AtomsPackTag1 tag;


    ~Ptensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1(){}

    Ptensors1(const AtomsPack& _atoms, const TENSOR& M):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors1(const AtomsPack& _atoms, const cnine::TensorView<TYPE>& M):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors1(const TENSOR& M, const AtomsPack& _atoms):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors1(const TENSOR& M, const AtomsPackTag1& _tag):
      BASE(_tag.obj->atoms.lock(),M),
      tag(_tag){}

    Ptensors1(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(_atoms,cnine::Gdims(_atoms.nrows1(),nc),0,_dev),
      tag(_atoms){}

    Ptensors1(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(_atoms,cnine::Gdims(_atoms.nrows1(),nc),fcode,_dev),
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
      BASE(_atoms),
      tag(_atoms){
      vparams v;
      unroller(v,args...);
      nc=v.nc;
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
      BASE(x.atoms,x.copy(_dev)), 
      //atoms(x.atoms),
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

    int size_of(const int i) const{
      return atoms.size_of(i);
    }

    int offset(const int i) const{
      return atoms.row_offset1(i);
    }

    Rtensor2_view view_of(const int i) const{
      return Rtensor2_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0],size_of(i),get_nc(),strides[0],strides[1],dev);
    }

    Rtensor2_view view_of(const int i, const int offs, const int m) const{
      return Rtensor2_view(const_cast<TYPE*>(get_arr())+offset(i)*strides[0]+offs*strides[1],
	size_of(i),m,strides[0],strides[1],dev);
    }

    Ptensor1view<TYPE> view_of(const int i, const vector<int>& ix) const{
      int nc=get_nc();
      return Ptensor1view<TYPE>(const_cast<TYPE*>(get_arr())+offset(i)*nc,nc,nc,1,ix,get_dev());
    }

    Ptensor1view<TYPE> view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      int nc=get_nc();
      return Ptensor1view<TYPE>(const_cast<TYPE*>(get_arr())+offset(i)*nc+offs,n,nc,1,ix,get_dev());
    }

    //TENSOR tensor_of(const int i) const{
    //return TENSOR::rows(offset(i),size_of(i));
    //}

    Ptensor1<TYPE> operator()(const int i) const{
      return Ptensor1(TENSOR(TENSOR::rows(offset(i),size_of(i))),atoms_of(i));
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
    static Ptensors1<TYPE> gather(const AtomsPack& a, const SOURCE& x, const int min_overlaps=1){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      Ptensors1<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,min_overlaps);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x, const int min_overlaps=1){
      //add_gather(x,ptens_global::overlaps_cache(atoms,x.atoms));
      add_gather(x,PtensorMapFactory::overlaps(atoms,x.atoms));
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      //add_gather_back(x,ptens_global::overlaps_cache(x.atoms,atoms));
      add_gather_back(x,PtensorMapFactory::overlaps(x.atoms,atoms));
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const PtensorMap& map){
      if(ptens_global::row_level_operations){
	rmap(x,map)(*this,x);
      }else{
	int nc=x.get_nc();
	if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	if constexpr(std::is_same<SOURCE,Ptensors1<TYPE> >::value){
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),nc);
	}
	if constexpr(std::is_same<SOURCE,Ptensors2<TYPE> >::value){
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),2*nc);
	}
      }
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const PtensorMap& map){
      if(ptens_global::row_level_operations){
	x.rmap(*this,map).inv()(*this,x);
      }else{
	int nc=get_nc();
	if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.out()),map.in());
	if constexpr(std::is_same<OUTPUT,Ptensors1<TYPE> >::value){
	  broadcast0(x.reduce0(map.atoms(),map.out(),0,nc),map.in());
	  broadcast1(x.reduce1(map.atoms(),map.out(),nc,nc),map.in());
	}
	if constexpr(std::is_same<OUTPUT,Ptensors2<TYPE> >::value){
	  broadcast0(x.reduce0_shrink(map.atoms(),map.out(),0,nc),map.in());
	  broadcast1(x.reduce1_shrink(map.atoms(),map.out(),2*nc,nc),map.in());
	}
      }
    }


  private:

    template<typename SOURCE>
    RowLevelMap& rmap(const SOURCE& x, const PtensorMap& tmap) const{
      return *ptens_global::rmap_cache(tag,x.tag,tmap.obj);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    TENSOR reduce0(const int offs=0, int nc=0) const{
      TimedFn T("Ptensors1","reduce0",*this);
      int N=size();
      int dev=get_dev();
      if(nc==0) nc=get_nc()-offs;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({N,nc},0,dev);
      Rtensor2_view r=R.view2();
      for(int i=0; i<N; i++)
	view_of(i,offs,nc).sum0_into(r.slice0(i));
      return R;
    }

    TENSOR reduce1() const{
      return *this;
    }

    void add_reduce0_to(const TENSOR& R, const int offs=0) const{
      PTENS_ASSRT(R.ndims()==2);
      PTENS_ASSRT(R.dim(0)==size());
      PTENS_CPUONLY();
      int N=size();
      int nc=R.dim(1);
      Rtensor2_view r=R.view2();
      for(int i=0; i<N; i++)
	view_of(i,offs,nc).sum0_into(r.slice0(i));
    }


  public: // ---- Indexed Reductions -------------------------------------------------------------------------


    Ptensors0<TYPE> reduce0(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
      TimedFn T("Ptensors1","reduce0",*this,list,list.count1*nc);
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors0<TYPE> R(_atoms,nc,0,get_dev());
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  view_of(list.tens(i),list.ix(i)).sum0_into(R.view_of(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }


    Ptensors1<TYPE> reduce1(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
      TimedFn T("Ptensors1","reduce1",*this,list,list.count1*nc);
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      Ptensors1<TYPE> R(_atoms,nc,0,get_dev());
      if(dev==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view_of(i)+=view_of(list.tens(i),list.ix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    void add_reduce1_to(const Ptensors1& R, const AindexPack& list, const int offs=0) const{
      TimedFn T("Ptensors1","reduce1",*this,list,list.count1*nc);
      int nc=R.get_nc();
      if(dev==0){
 	int N=list.size();
 	for(int i=0; i<N; i++){
 	  if(list.nix(i)==0) continue;
 	  R.view_of(i)+=view_of(list.tens(i),list.ix(i));
 	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,list,0,nc,stream)));
    }

    void add_reduce0_to(const Ptensors0<TYPE>& R, const AindexPack& list, const int offs=0) const{
      TimedFn T("Ptensors1","reduce0",*this,list,list.count1*nc);
      int nc=R.get_nc();
      if(dev==0){
 	int N=list.size();
 	for(int i=0; i<N; i++){
 	  if(list.nix(i)==0) continue;
 	  view_of(list.tens(i),list.ix(i)).sum0_into(R.view_of(i));
 	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,list,0,nc,stream)));
    }

    
  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const TENSOR& X, const int offs=0){
      TimedFn T("Ptensors1","broadcast0",*this);
      int N=size();
      int nc=X.dim(1);
      PTENS_ASSRT(X.dim(0)==N);
      Rtensor2_view x=X.view2();
      for(int i=0; i<N; i++)
	view_of(i,offs,nc)+=cnine::repeat0(x.slice0(i),size_of(i));
    }

    void broadcast1(const TENSOR& X, const int offs=0){
      TimedFn T("Ptensors1","broadcast1",*this);
      int nc=X.dim(1);
      BASE::view2().block(0,offs,dim(0),nc)+=X.view2();
    }


  public: // ---- Indexed broadcasting -----------------------------------------------------------------------


    void broadcast0(const Ptensors0<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors1","brcast0",*this,x,list,list.count1*x.get_nc());
      if(dev==0){
	int N=list.size();
	const int nc=x.get_nc();
	for(int i=0; i<N; i++){
	  view_of(list.tens(i),list.ix(i),offs,nc)+=repeat0(x.view_of(i),list.nix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,list,offs,stream)));
    }

    void broadcast1(const Ptensors1<TYPE>& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors1","brcast1",*this,x,list,list.count1*x.get_nc());
      if(dev==0){
	int N=list.size();
	const int nc=x.get_nc();
	for(int i=0; i<N; i++){
	  if(x.size_of(i)==0) continue;
	  view_of(list.tens(i),list.ix(i),offs,nc)+=x.view_of(i);
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,list,offs,stream)));
    }
    

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
	oss<<(*this)(i).str(indent);
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
    //TENSOR reduce0() const{
    //TimedFn T("Ptensors1","reduce0",*this);
    //int N=size();
    //int dev=get_dev();
    //cnine::using_vram_manager vv(ptens_global::vram_manager);
    //TENSOR R({N,get_nc()},0,dev);
    //Rtensor2_view r=R.view2();
    //for(int i=0; i<N; i++)
    //view_of(i).sum0_into(r.slice0(i));
    //return R;
    //}

