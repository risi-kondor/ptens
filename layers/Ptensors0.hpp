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
//#include "PtensLoggedTimer.hpp"
#include "Ltensor.hpp"
#include "Ptensorsb.hpp"
#include "AtomsPackTag.hpp"


namespace ptens{



  template<typename TYPE>
  class Ptensors0: public Ptensorsb<TYPE>, public cnine::diff_class<Ptensors0<TYPE> >{
  public:

    friend class Ptensors1<TYPE>;
    friend class Ptensors2<TYPE>;

    typedef Ptensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    using cnine::diff_class<Ptensors0<TYPE> >::grad;
    using TENSOR::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;


    AtomsPack atoms;
    AtomsPackTag0 tag;


    ~Ptensors0(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0(){}

    Ptensors0(const TENSOR& M, const AtomsPack& _atoms):
      BASE(M),
      atoms(_atoms),
      tag(_atoms){}

    Ptensors0(const TENSOR& M, const AtomsPackTag0& _tag):
      BASE(M),
      atoms(_tag.obj->atoms.lock()),
      tag(_tag){}

    Ptensors0(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.size(),nc),0,_dev),
      atoms(_atoms),
      tag(_atoms){}

    Ptensors0(const AtomsPack& _atoms, const int nc, const int fcode, const int _dev):
      BASE(cnine::Gdims(_atoms.size(),nc),fcode,_dev),
      atoms(_atoms),
      tag(_atoms){}


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
      atoms(_atoms),
      tag(_atoms){
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


    //Ptensors0(const TENSOR& x, const AtomsPack& _atoms):
    //BASE(x),
    //atoms(_atoms){}
    

  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms),
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

    int size() const{
      return atoms.size();
    }

    int get_nc() const{
      return TENSOR::dim(1);
    }

    const AtomsPack& get_atoms() const{
      return atoms;
    }

    int offset(const int i) const{
      return atoms.row_offset0(i);
    }

    int index_of(const int i) const{
      return i;
    }

    Atoms atoms_of(const int i) const{
      return atoms(i);
    }
    
    TENSOR tensor_of(const int i) const{
      return TENSOR::row(offset(i));
    }

    Ptensor0<TYPE> operator()(const int i) const{
      return Ptensor0(tensor_of(i).view1(),atoms_of(i));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0<float> linmaps(const SOURCE& x){
      Ptensors0<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
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


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0<TYPE> gather(const SOURCE& x, const AtomsPack& a){
      int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
      Ptensors0<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x){
      auto overlaps=ptens_global::overlaps_cache(atoms,x.atoms);
      if(ptens_global::row_level_operations){
	rmap(x,overlaps)(*this,x);
      }else{
      }
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      auto overlaps=ptens_global::overlaps_cache(x.atoms,atoms);
      if(ptens_global::row_level_operations){
	x.rmap(*this,overlaps).inv()(*this,x);
      }else{
      }
    }

    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){
      auto overlaps=ptens_global::overlaps_cache(x.atoms,atoms);
      if(ptens_global::row_level_operations){
	x.rmap(*this,overlaps).inv()(get_grad(),x.get_grad());
      }else{
      }
    }


  private:

    template<typename SOURCE>
    RowLevelMap& rmap(const SOURCE& x, const shared_ptr<TensorLevelMapObj>& tmap) const{
      return *ptens_global::rmap_cache(tag,x.tag,tmap);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      return *this;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& X, const int offs=0){
      int nc=X.dim(1);
      BASE::view2().cols(offs,nc)+=X.view2();
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



  template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
  inline Ptensors0<float> linmaps0(const SOURCE& x){
    Ptensors0<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
  Ptensors0<float> gather0(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
    Ptensors0<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }

}


#endif 

