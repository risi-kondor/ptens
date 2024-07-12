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

#ifndef _ptens_Ptensors0
#define _ptens_Ptensors0

#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "Ptensor0.hpp"
#include "Ptensors.hpp"
#include "AtomsPackTag.hpp"
#include "TensorLevelMap.hpp"


namespace ptens{



  template<typename TYPE>
  class Ptensors0: public Ptensors<TYPE>, public cnine::diff_class<Ptensors0<TYPE> >{
  public:

    friend class Ptensors1<TYPE>;
    friend class Ptensors2<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;

    using cnine::diff_class<Ptensors0<TYPE> >::grad;
    using TENSOR::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;
    using TENSOR::get_arr;

    using BASE::nc;
    using BASE::atoms;
    using BASE::size;
    using BASE::atoms_of;
    using BASE::get_nc;

    AtomsPackTag0 tag;


    ~Ptensors0(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0(){}

    Ptensors0(const AtomsPack& _atoms, const TENSOR& M):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors0(const TENSOR& M, const AtomsPack& _atoms):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors0(const TENSOR& M, const AtomsPackTag0& _tag):
      BASE(_tag.obj->atoms.lock(),M),
      tag(_tag){}

    Ptensors0(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(_atoms,cnine::Gdims(_atoms.size(),nc),0,_dev),
      tag(_atoms){}

    Ptensors0(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(_atoms,cnine::Gdims(_atoms.size(),nc),fcode,_dev),
      tag(_atoms){}

    //Ptensors0(const int N, const int nc, const int fcode, const int _dev):
    //BASE(cnine::Gdims(N,nc),fcode,_dev){}


    static Ptensors0 cat(const vector<Ptensors0>& list){
      vector<AtomsPack> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      if(ptens_global::cache_atomspack_cats) 
	return Ptensors0(cnine::Ltensor<TYPE>::stack(0,list),ptens_global::atomspack_cat_cache(v));
      return Ptensors0(cnine::Ltensor<TYPE>::stack(0,list),AtomsPack::cat(v));
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors0(const AtomsPack& _atoms, const Args&... args):
      BASE(_atoms),
      tag(_atoms){
      vparams v;
      unroller(v,args...);
      nc=v.nc;
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


    Ptensors0 copy() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors0(TENSOR::copy(),tag);
    }

    Ptensors0 copy(const int _dev) const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors0(TENSOR::copy(_dev),tag);
    }

    Ptensors0 zeros_like() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors0(TENSOR::zeros_like(),atoms);
    }

    Ptensors0 gaussian_like() const{
      return Ptensors0(BASE::gaussian_like(),atoms);
    }

    static Ptensors0 zeros_like(const Ptensors0& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors0(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors0 zeros_like(const Ptensors0& x, const int nc){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors0(TENSOR({x.dim(0),nc},0,x.get_dev()),x.atoms);
    }

    static Ptensors0 gaussian_like(const Ptensors0& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return Ptensors0(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors0* new_zeros_like(const Ptensors0& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new Ptensors0(x.TENSOR::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


#ifdef _WITH_ATEN
    //    static Ptensors0 view(const Atoms& _atoms, const at::Tensor& x){
      // Check dimensions of x here!
      //return Ptensors0(_atoms,BASE::view(x));
    //}
#endif 

    //Ptensors0(const TENSOR& x, const AtomsPack& _atoms):
    //BASE(x),
    //atoms(_atoms){}
    

  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x, const int _dev):
      BASE(x.atoms,x.copy(_dev)), 
      //atoms(x.atoms),
      tag(x.tag){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors0& get_grad(){
      return cnine::diff_class<Ptensors0<TYPE> >::get_grad();
    }

    const Ptensors0& get_grad() const{
      return cnine::diff_class<Ptensors0<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 0;
    }

    //int size() const{
    //return atoms.size();
    //}

    //int get_nc() const{
    //return TENSOR::dim(1);
    //}

    //const AtomsPack& get_atoms() const{
    //return atoms;
    //}

    int offset(const int i) const{
      return i; 
    }

    int index_of(const int i) const{
      return i;
    }

    //Atoms atoms_of(const int i) const{
    //return atoms(i);
    //}
    
    int size_of(const int i) const{
      return 1;
    }

    TENSOR tensor_of(const int i) const{
      return TENSOR::row(offset(i));
    }

    Rtensor1_view view_of(const int i) const{
      return Rtensor1_view(const_cast<float*>(get_arr())+get_nc()*i,get_nc(),1,get_dev());
    }

    Rtensor1_view view_of(const int i, const int offs, const int n) const{
      return Rtensor1_view(const_cast<float*>(get_arr())+get_nc()*i+offs,n,1,get_dev());
    }

    Ptensor0<TYPE> operator()(const int i) const{
      return Ptensor0(tensor_of(i).view1(),atoms_of(i));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensors<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0<float> linmaps(const SOURCE& x){
      Ptensors0<float> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      add(x);
    }

    void add_linmaps(const Ptensors1<TYPE>& x){
      add(x.reduce0());
    }

    void add_linmaps(const Ptensors2<TYPE>& x){
      add(x.reduce0());
    }

    void add_linmaps_back(const Ptensors0<TYPE>& r){
      add(r);
    }

    void add_linmaps_back(const Ptensors1<TYPE>& r){
      add(r.reduce0());
    }

    void add_linmaps_back(const Ptensors2<TYPE>& r){
      int nc=get_nc();
      add(r.reduce0_shrink(0,nc));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensors<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0<TYPE> gather(const AtomsPack& a, const SOURCE& x){
      int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
      Ptensors0<TYPE> R(a,nc,x.get_dev());
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
	if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	if constexpr(std::is_same<SOURCE,Ptensors1<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	if constexpr(std::is_same<SOURCE,Ptensors2<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
      }
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const TensorLevelMap& map){
      if(ptens_global::row_level_operations){
	x.rmap(*this,map).inv()(*this,x);
      }else{
	if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.out()),map.in());
	if constexpr(std::is_same<OUTPUT,Ptensors1<TYPE> >::value)
	  broadcast0(x.reduce0(map.atoms(),map.out()),map.in());
	if constexpr(std::is_same<OUTPUT,Ptensors2<TYPE> >::value)
	  broadcast0(x.reduce0_shrink(map.atoms(),map.out()),map.in());
      }
    }


  private:

    template<typename SOURCE>
    RowLevelMap& rmap(const SOURCE& x, const TensorLevelMap& tmap) const{
      return *ptens_global::rmap_cache(tag,x.tag,tmap.obj);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      return *this;
    }

    Ptensors0 reduce0(const AtomsPack& _atoms, const AindexPack& list) const{
      TimedFn T("Ptensors0","reduce0",*this,list,list.size()*get_nc());
      Ptensors0 R(_atoms,get_nc(),0,get_dev());
      if(get_dev()==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view_of(i)=view_of(list.tix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,list,0,nc,stream)));
      return R;
    }

    void add_reduce0_to(const Ptensors0& R, const AindexPack& list) const{
      TimedFn T("Ptensors0","reduce0",*this,list,list.size()*get_nc());
      if(get_dev()==0){
	int N=list.size();
	for(int i=0; i<N; i++){
	  if(list.nix(i)==0) continue;
	  R.view_of(i)=view_of(list.tix(i));
	}
      }
      GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,list,0,nc,stream)));
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      int nc=X.dim(1);
      BASE::view2().cols(offs,nc)+=X.view2();
    }

    void broadcast0(const Ptensors0& x, const AindexPack& list, const int offs=0){
      TimedFn T("Ptensors0","broadcast0",*this,x,list,list.size()*get_nc());
      if(get_dev()==0){
	int N=list.size();
	const int n=x.get_nc();
	for(int i=0; i<N; i++)
	  view_of(list.tix(i),offs,n)+=x.view_of(i);
      }
      GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,list,offs,stream)));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors0";
    }

    string repr() const{
      return "<Ptensors0[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors0 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors0& x){
      stream<<x.str(); return stream;}


  };



  template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensors<float>, SOURCE>::value, SOURCE>::type>
  inline Ptensors0<float> linmaps0(const SOURCE& x){
    Ptensors0<float> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensors<float>, SOURCE>::value, SOURCE>::type>
  Ptensors0<float> gather0(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
    Ptensors0<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }

}


#endif 
