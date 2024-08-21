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
#include "GatherPlanFactory.hpp"


namespace ptens{

  #ifdef _WITH_CUDA
  extern void Ptensors1_reduce0_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_reduce1_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors1_broadcast0_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  extern void Ptensors1_broadcast1_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  #endif 


  template<typename TYPE>
  class Ptensors1: public Ptensors<TYPE>, public cnine::diff_class<Ptensors1<TYPE> >{
  public:

    friend class Ptensors0<TYPE>;
    friend class Ptensors2<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    //typedef cnine::Ltensor<TYPE> TENSOR;
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

    //Ptensors1(const AtomsPack& _atoms, const cnine::TensorView<TYPE>& M):
    //BASE(_atoms,M),
    //tag(_atoms){}

    Ptensors1(const TENSOR& M, const AtomsPack& _atoms):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors1(const TENSOR& M, const AtomsPackTag1& _tag):
      BASE(_tag.obj->atoms.lock(),M),
      tag(_tag){}

    Ptensors1(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(_atoms,cnine::Gdims({_atoms.nrows1(),nc}),0,_dev),
      tag(_atoms){}

    Ptensors1(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(_atoms,cnine::Gdims({_atoms.nrows1(),nc}),fcode,_dev),
      tag(_atoms){}


    static Ptensors1 cat(const vector<Ptensors1>& list){
      vector<AtomsPack> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      if(ptens_global::cache_atomspack_cats) 
	return Ptensors1(TENSOR::stack(0,list),ptens_global::atomspack_cat_cache(v));
      return Ptensors1(TENSOR::stack(0,list),AtomsPack::cat(v));
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
      return Ptensors1(TENSOR::copy(),atoms);
    }

    Ptensors1 copy(const int _dev) const{
      return Ptensors1(TENSOR::copy(_dev),atoms);
    }

    Ptensors1 zeros_like() const{
      return Ptensors1(TENSOR::zeros_like(),atoms);
    }

    Ptensors1 zeros_like(const int nc) const{
      return Ptensors1(TENSOR({dim(0),nc},0,get_dev()),atoms);
    }

    Ptensors1 gaussian_like() const{
      return Ptensors1(BASE::gaussian_like(),atoms);
    }

    static Ptensors1 zeros_like(const Ptensors1& x){
      return Ptensors1(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors1 gaussian_like(const Ptensors1& x){
      return Ptensors1(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors1* new_zeros_like(const Ptensors1& x){
      return new Ptensors1(x.BASE::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------




  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors1(const Ptensors1& x, const int _dev):
      BASE(x.atoms,x.copy(_dev)), 
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

    Ptensor1<TYPE> operator()(const int i) const{
      return Ptensor1(TENSOR(TENSOR::rows(offset(i),size_of(i))),atoms_of(i));
    }

    TENSOR view3(int offs=0, int nc=0) const{
      PTENS_ASSRT(atoms.constk()>0);
      if(nc==0) nc=dim(1);
      return cols(offs,nc).split(0,atoms.constk());
    }

    //const cnine::Rtensor3_view view3(const int K) const{
    //int nc=get_nc();
    //return cnine::Rtensor3_view(const_cast<float*>(get_arr()),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
    //}

//     cnine::Rtensor3_view view3(const int K){
//       int nc=get_nc();
//       return cnine::Rtensor3_view(get_arr(),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
//     }

//     const cnine::Rtensor3_view view3(const int K, const int offs, const int nc) const{
//       int _nc=get_nc();
//       return cnine::Rtensor3_view(const_cast<float*>(get_arr())+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
//     }

//     cnine::Rtensor3_view view3(const int K, const int offs, const int nc){
//       int _nc=get_nc();
//       return cnine::Rtensor3_view(get_arr()+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
//     }


    Ptensor1view<TYPE> view_of(const AindexPackB& apack, const int i) const{
      int nc=get_nc();
      return Ptensor1view<TYPE>(const_cast<float*>(get_arr())+apack.soffset(i)*nc,nc,nc,1,apack.ix(i),get_dev());
    }

    Ptensor1view<TYPE> view_of(const AindexPackB& apack, const int i, const int offs, const int n) const{
      int nc=get_nc();
      return Ptensor1view<TYPE>(const_cast<float*>(get_arr())+apack.soffset(i)*nc+offs,n,nc,1,apack.ix(i),get_dev());
    }

    void zip0(const AindexPackB& map, const TENSOR& M, 
      const std::function<void(const Rtensor1_view&,const Ptensor1view<TYPE>&, int)>& lambda, const int offset=0, int n=0) const{
      int N=map.size();
      int nc=get_nc();
      if(n==0) n=nc-offset; 
      for(int i=0; i<N; i++)
	lambda(M.row(map.toffset(i)).view1(),
	  Ptensor1view<TYPE>(const_cast<float*>(get_arr())+map.soffset(i)*nc+offset,
	    n,nc,1,map.ix(i),get_dev()),map.nix(i));
    }

    void zip1(const AindexPackB& map, const TENSOR& M, 
      const std::function<void(const Rtensor2_view&, const Ptensor1view<TYPE>&, int)>& lambda, const int offset=0, int n=0) const{
      int N=map.size();
      int nc=get_nc();
      if(n==0) n=nc-offset; 
      for(int i=0; i<N; i++)
	lambda(M.rows(map.toffset(i),map.nix(i)).view2(),
	  Ptensor1view<TYPE>(const_cast<float*>(get_arr())+map.soffset(i)*nc+offset,
	    n,nc,1,map.ix(i),get_dev()),map.nix(i));
    }


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
    static Ptensors1<TYPE> gather(const AtomsPack& atoms, const SOURCE& x){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      Ptensors1<TYPE> R(atoms,nc,x.get_dev());
      R.add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
      return R;
    }

    template<typename SOURCE>
    static Ptensors1<TYPE> gather(const AtomsPack& a, const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
      Ptensors1<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,map);
      return R;
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc();

      if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value){
	auto pmap=GatherPlanFactory::gather_map0(map,atoms,x.atoms,1,x.getk());
	broadcast0(x.reduce0(pmap.in()),pmap.out(),0);
      }

      if constexpr(std::is_same<SOURCE,Ptensors1<TYPE> >::value){
	auto pmap0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,1,x.getk());
	auto pmap1=GatherPlanFactory::gather_map1(map,atoms,x.atoms,1,x.getk());
	broadcast0(x.reduce0(pmap0.in()),pmap0.out(),0);
	broadcast1(x.reduce1(pmap1.in()),pmap1.out(),nc);
      }

      if constexpr(std::is_same<SOURCE,Ptensors2<TYPE> >::value){
	auto pmap0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,1,x.getk());
	auto pmap1=GatherPlanFactory::gather_map1(map,atoms,x.atoms,1,x.getk());
	broadcast0(x.reduce0(pmap0.in()),pmap0.out(),0);
	broadcast1(x.reduce1(pmap1.in()),pmap1.out(),2*nc);
      }

    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const LayerMap& map){
      int nc=get_nc();

      if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value){
	auto pmap=GatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),1);
	broadcast0(x.reduce0(pmap.out()),pmap.in(),0);
      }

      if constexpr(std::is_same<OUTPUT,Ptensors1<TYPE> >::value){
	auto pmap0=GatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),1);
	auto pmap1=GatherPlanFactory::gather_map1(map,x.atoms,atoms,x.getk(),1);
	broadcast0(x.reduce0(pmap0.out(),0,nc),pmap0.in(),0);
	broadcast1(x.reduce1(pmap1.out(),nc,nc),pmap1.in(),0);
      }

      if constexpr(std::is_same<OUTPUT,Ptensors2<TYPE> >::value){
	auto pmap0=GatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),1);
	auto pmap1=GatherPlanFactory::gather_map1(map,x.atoms,atoms,x.getk(),1);
	broadcast0(x.reduce0_shrink(pmap0.out(),0,nc),pmap0.in(),0);
	broadcast1(x.reduce1_shrink(pmap1.out(),2*nc,nc),pmap1.in(),2*nc);
      }

    }
    

#include "Ptensors1_reductions.hpp"
#include "Ptensors1_broadcasting.hpp"


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

