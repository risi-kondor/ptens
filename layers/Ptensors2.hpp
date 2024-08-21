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
#include "GatherPlanFactory.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors0;
  template<typename TYPE> class Ptensors1;

  #ifdef _WITH_CUDA
  extern void Ptensors2_reduce0_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors2_reduce0_shrink_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors2_reduce1_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors2_reduce1_shrink_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors2_reduce2_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors2_reduce2_shrink_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    int offs, int n, const cudaStream_t& stream);
  extern void Ptensors2_broadcast0_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  extern void Ptensors2_broadcast0_shrink_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  extern void Ptensors2_broadcast1_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  extern void Ptensors2_broadcast1_shrink_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  extern void Ptensors2_broadcast2_cu(const cnine::Ltensor<float>& r, const cnine::Ltensor<float>& x, const AindexPackB& map, 
    const int offs, const cudaStream_t& stream);
  #endif 


  template<typename TYPE>
  class Ptensors2: public Ptensors<TYPE>, public cnine::diff_class<Ptensors2<TYPE> >{
  public:

    friend class Ptensors0<TYPE>;
    friend class Ptensors1<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    //typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor1_view Rtensor1_view;
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

    //Ptensors2(const AtomsPack& _atoms, const cnine::TensorView<TYPE>& M):
    //BASE(_atoms,M),
    //tag(_atoms){}

    Ptensors2(const TENSOR& M, const AtomsPack& _atoms):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors2(const TENSOR& M, const AtomsPackTag2& _tag):
      BASE(_tag.obj->atoms.lock(),M),
      tag(_tag){}

    Ptensors2(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(_atoms,cnine::Gdims({_atoms.nrows2(),nc}),0,_dev),
      tag(_atoms){}

    Ptensors2(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(_atoms,cnine::Gdims({_atoms.nrows2(),nc}),fcode,_dev),
      tag(_atoms){}


    static Ptensors2 cat(const vector<Ptensors2>& list){
      vector<AtomsPack> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      if(ptens_global::cache_atomspack_cats) 
	return Ptensors2(TENSOR::stack(0,list),ptens_global::atomspack_cat_cache(v));
      return Ptensors2(TENSOR::stack(0,list),AtomsPack::cat(v));
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

    int size_of(const int i) const{
      return atoms.size_of(i);
    }

    int offset(const int i) const{
      return atoms.row_offset2(i);
    }

    int offset1(const int i) const{
      return atoms.row_offset1(i);
    }

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

    Ptensor2view<TYPE> view_of(const AindexPackB& apack, const int i, const vector<int>& ix) const{
      int nc=get_nc();
      int k=apack.ssize(i);
      return Ptensor2view<TYPE>(const_cast<TYPE*>(get_arr())+apack.soffset(i)*nc,nc,k*nc,nc,1,apack.ix(i),get_dev());
    }

    Ptensor2view<TYPE> view_of(const AindexPackB& apack, const int i, const vector<int>& ix, const int offs, const int n) const{
      int nc=get_nc();
      int k=apack.ssize(i);
      return Ptensor2view<TYPE>(const_cast<TYPE*>(get_arr())+apack.soffset(i)*nc+offs,n,k*nc,nc,1,apack.ix(i),get_dev());
    }

    TENSOR view4(int offs=0, int nc=0) const{
      PTENS_ASSRT(atoms.constk()>0);
      if(nc==0) nc=dim(1);
      return cols(offs,nc).split(0,atoms.constk()).split(0,atoms.constk());
    }

    void zip0(const AindexPackB& map, const TENSOR& M, 
      const std::function<void(const Rtensor1_view&,const Ptensor2view<TYPE>&, int)>& lambda, const int offset=0, int n=0) const{
      int N=map.size();
      int nc=get_nc();
      if(n==0) n=nc-offset; 
      for(int i=0; i<N; i++)
	lambda(M.row(map.toffset(i)).view1(),
	  Ptensor2view<TYPE>(const_cast<float*>(get_arr())+map.soffset(i)*nc+offset,
	    n,map.ssize(i)*nc,nc,1,map.ix(i),get_dev()),map.nix(i));
    }

    void zip1(const AindexPackB& map, const TENSOR& M, 
      const std::function<void(const Rtensor2_view&, const Ptensor2view<TYPE>&, int)>& lambda, const int offset=0, int n=0) const{
      int N=map.size();
      int nc=get_nc();
      if(n==0) n=nc-offset; 
      for(int i=0; i<N; i++)
	lambda(M.rows(map.toffset(i),map.nix(i)).view2(),
	  Ptensor2view<TYPE>(const_cast<float*>(get_arr())+map.soffset(i)*nc+offset,
	    n,map.ssize(i)*nc,nc,1,map.ix(i),get_dev()),map.nix(i));
    }

    void zip2(const AindexPackB& map, const TENSOR& M, 
      const std::function<void(const Rtensor3_view&, const Ptensor2view<TYPE>&, int)>& lambda, const int offset=0, int n=0) const{
      int N=map.size();
      int nc=get_nc();
      if(n==0) n=nc-offset; 
      for(int i=0; i<N; i++){
	int k=map.nix(i);
	lambda(cnine::split0(M.rows(map.toffset(i),k*k).view2(),k,k),
	  Ptensor2view<TYPE>(const_cast<float*>(get_arr())+map.soffset(i)*nc+offset,
	    n,map.ssize(i)*nc,nc,1,map.ix(i),get_dev()),map.nix(i));
      }
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
      R.add_gather(x,LayerMap::overlaps_map(a,x.atoms));
      return R;
    }

    template<typename SOURCE>
    static Ptensors2<TYPE> gather(const AtomsPack& a, const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
      Ptensors2<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,map);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc();

      if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value){
	auto plan=GatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0(x.reduce0(plan.in()),plan.out(),0);
      }

      if constexpr(std::is_same<SOURCE,Ptensors1<TYPE> >::value){
	auto plan0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0(x.reduce0(plan0.in()),plan0.out(),0);
	auto plan1=GatherPlanFactory::gather_map1(map,atoms,x.atoms,2,x.getk());
	broadcast1(x.reduce1(plan1.in()),plan1.out(),2*nc);
      }

      if constexpr(std::is_same<SOURCE,Ptensors2<TYPE> >::value){
	auto plan0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0(x.reduce0(plan0.in()),plan0.out(),0);
	auto plan1=GatherPlanFactory::gather_map1(map,atoms,x.atoms,2,x.getk());
	broadcast1(x.reduce1(plan1.in()),plan1.out(),4*nc);
	auto plan2=GatherPlanFactory::gather_map2(map,atoms,x.atoms,2,x.getk());
	broadcast2(x.reduce2(plan2.in()),plan2.out(),13*nc);
      }

    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const LayerMap& map){
      int nc=get_nc();

      if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value){
	auto plan0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0_shrink(x.reduce0(plan0.out()),plan0.in());
      }

      if constexpr(std::is_same<OUTPUT,Ptensors1<TYPE> >::value){
	auto plan0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0_shrink(x.reduce0(plan0.out(),0,2*nc),plan0.in());
	auto plan1=GatherPlanFactory::gather_map1(map,atoms,x.atoms,2,x.getk());
	broadcast1_shrink(x.reduce1(plan1.out(),2*nc,3*nc),plan1.in());
      }

      if constexpr(std::is_same<OUTPUT,Ptensors2<TYPE> >::value){
	auto plan0=GatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0_shrink(x.reduce0_shrink(plan0.out(),0,2*nc),plan0.in());
	auto plan1=GatherPlanFactory::gather_map1(map,atoms,x.atoms,2,x.getk());
	broadcast1_shrink(x.reduce1_shrink(plan1.out(),4*nc,3*nc),plan1.in());
	auto plan2=GatherPlanFactory::gather_map2(map,atoms,x.atoms,2,x.getk());
	broadcast2(x.reduce2_shrink(plan2.out(),13*nc,nc),plan2.in());
      }

    }


  private:

    #include "Ptensors2_reductions.hpp"
    #include "Ptensors2_broadcasting.hpp"


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
	oss<<(*this)(i).str(indent);
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
    //template<typename SOURCE>
    //RowLevelMap& rmap(const SOURCE& x, const PtensorMap& tmap) const{
    //return *ptens_global::rmap_cache(tag,x.tag,tmap.obj);
    //}

