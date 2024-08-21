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

#include "Ptensor0.hpp"
#include "Ptensors.hpp"
#include "AtomsPackTag.hpp"
#include "GatherPlanFactory.hpp"


namespace ptens{

  #ifdef _WITH_CUDA 
  extern void Ptensors0_reduce0_cu(const cnine::Ltensor<float>& R, const cnine::Ltensor<float>& x, 
    const AindexPackB& map, int offs, int n, const cudaStream_t& stream);
  extern void Ptensors0_broadcast0_cu(const cnine::Ltensor<float>& x, const cnine::Ltensor<float>& R, 
    const AindexPackB& map, const int offs, const cudaStream_t& stream);
  #endif 


  template<typename TYPE>
  class Ptensors0: public Ptensors<TYPE>, public cnine::diff_class<Ptensors0<TYPE> >{
  public:

    friend class Ptensors1<TYPE>;
    friend class Ptensors2<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    //typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor1_view Rtensor1_view;

    using cnine::diff_class<Ptensors0<TYPE> >::grad;
    using TENSOR::get_dev;
    using TENSOR::dim;
    using TENSOR::dev;
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

    //Ptensors0(const AtomsPack& _atoms, const cnine::TensorView<TYPE>& M):
    //BASE(_atoms,M),
    //tag(_atoms){}

    Ptensors0(const TENSOR& M, const AtomsPack& _atoms):
      BASE(_atoms,M),
      tag(_atoms){}

    Ptensors0(const TENSOR& M, const AtomsPackTag0& _tag):
      BASE(_tag.obj->atoms.lock(),M),
      tag(_tag){}

    Ptensors0(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(_atoms,cnine::Gdims({_atoms.size(),nc}),0,_dev),
      tag(_atoms){}

    Ptensors0(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(_atoms,cnine::Gdims({_atoms.size(),nc}),fcode,_dev),
      tag(_atoms){}


    /*
    static Ptensors0 cat(const vector<Ptensors0>& list){
      vector<AtomsPack> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      if(ptens_global::cache_atomspack_cats) 
	return Ptensors0(TENSOR::stack(0,list),ptens_global::atomspack_cat_cache(v));
      return Ptensors0(TENSOR::stack(0,list),AtomsPack::cat(v));
    }
    */

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
      return Ptensors0(TENSOR::copy(),tag);
    }

    Ptensors0 copy(const int _dev) const{
      return Ptensors0(TENSOR::copy(_dev),tag);
    }

    Ptensors0 zeros_like() const{
      return Ptensors0(TENSOR::zeros_like(),atoms);
    }

    Ptensors0 gaussian_like() const{
      return Ptensors0(BASE::gaussian_like(),atoms);
    }

    static Ptensors0 zeros_like(const Ptensors0& x){
      return Ptensors0(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors0 zeros_like(const Ptensors0& x, const int nc){
      return Ptensors0(TENSOR({x.dim(0),nc},0,x.get_dev()),x.atoms);
    }

    static Ptensors0 gaussian_like(const Ptensors0& x){
      return Ptensors0(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors0* new_zeros_like(const Ptensors0& x){
      return new Ptensors0(x.TENSOR::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x, const int _dev):
      BASE(x.atoms,x.copy(_dev)), 
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

    int offset(const int i) const{
      return i; 
    }

    int index_of(const int i) const{
      return i;
    }

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


    Rtensor1_view view_of(const AindexPackB& apack, const int i) const{
      return Rtensor1_view(const_cast<float*>(get_arr())+get_nc()*apack.soffset(i),get_nc(),1,get_dev());
    }

    Rtensor1_view view_of(const AindexPackB& apack, const int i, const int offs, const int n) const{
      return Rtensor1_view(const_cast<float*>(get_arr())+get_nc()*apack.soffset(i)+offs,n,1,get_dev());
    }

    void zip0(const AindexPackB& map, const TENSOR& M, 
      const std::function<void(const Rtensor1_view&, const Rtensor1_view&, int)>& lambda, const int offset=0, int n=0) const{
      int N=map.size();
      int nc=get_nc();
      if(n==0) n=nc-offset; 
      for(int i=0; i<N; i++)
	lambda(M.row(map.toffset(i)).view1(),
	  Rtensor1_view(const_cast<float*>(get_arr())+map.soffset(i)*nc+offset,n,1,get_dev()),map.nix(i));
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
      add(r.reduce0_shrink(0,get_nc()));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensors<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0<TYPE> gather(const AtomsPack& atoms, const SOURCE& x){
      int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
      Ptensors0<TYPE> R(atoms,nc,x.get_dev());
      R.add_gather(x,LayerMap::overlaps_map(atoms,x.atoms));
      return R;
    }

   template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensors<float>, SOURCE>::value, SOURCE>::type>
   static Ptensors0<TYPE> gather(const AtomsPack& a, const SOURCE& x, const LayerMap& map){
      int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
      Ptensors0<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x,map);
      return R;
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const LayerMap& map){
      auto plan=GatherPlanFactory::gather_map0(map,atoms,x.atoms,0,x.getk());
      if constexpr(std::is_same<SOURCE,Ptensors0<TYPE> >::value)
	broadcast0(x.reduce0(plan.in()),plan.out(),0);
      if constexpr(std::is_same<SOURCE,Ptensors1<TYPE> >::value)
	broadcast0(x.reduce0(plan.in()),plan.out(),0);
      if constexpr(std::is_same<SOURCE,Ptensors2<TYPE> >::value)
	broadcast0(x.reduce0(plan.in()),plan.out(),0);
     }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const LayerMap& map){
      auto plan=GatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),0);
      if constexpr(std::is_same<OUTPUT,Ptensors0<TYPE> >::value)
	broadcast0(x.reduce0(plan.out()),plan.in(),0);
      if constexpr(std::is_same<OUTPUT,Ptensors1<TYPE> >::value)
	broadcast0(x.reduce0(plan.out()),plan.in(),0);
      if constexpr(std::is_same<OUTPUT,Ptensors2<TYPE> >::value)
	broadcast0(x.reduce0_shrink(plan.out()),plan.in(),0);
    }


#include "Ptensors0_reductions.hpp"
#include "Ptensors0_broadcasting.hpp"


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
	oss<<(*this)(i).str(indent);
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
