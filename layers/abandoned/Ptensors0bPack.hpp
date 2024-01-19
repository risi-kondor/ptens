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

#ifndef _ptens_Ptensors0bPack
#define _ptens_Ptensors0bPack

#include "diff_class.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"

#include "AtomsPack0.hpp"
#include "Ptensorsb.hpp"
#include "Ptensors0b.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors1bPack;
  template<typename TYPE> class Ptensors2bPack;


  template<typename TYPE>
  class Ptensors0bPack: public Ptensorsb<TYPE>, public cnine::diff_class<Ptensors0bPack<TYPE> >{

  public:

    typedef Ptensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    using cnine::diff_class<Ptensors0bPack<TYPE> >::grad;
    using TENSOR::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;


    AtomsPack0pack atoms;


    ~Ptensors0bPack(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0bPack(){}

    Ptensors0bPack(const TENSOR& x, const vector<AtomsPack0>& _atoms):
      BASE(x),
      atoms(_atoms){}
    
    Ptensors0bPack(const AtomsPack0pack& _atoms, const int nc, const int fcode, const int _dev):
      BASE({_atoms.size0(),nc},fcode,_dev),
      atoms(_atoms){}


    /*

    Ptensors0bPack(const TENSOR& M): 
      BASE(M.copy()),
      atoms(AtomsPack0(M.dim(0))){}

    Ptensors0bPack(const AtomsPack0& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors0bPack(const AtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()),
      atoms(_atoms){}

    Ptensors0bPack(const AtomsPack& _atoms, const int nc, const int _dev=0):
      BASE(cnine::Gdims(_atoms.size(),nc),0,_dev),
      atoms(_atoms){}

    Ptensors0bPack(const int n, const int nc, const int fcode=0, const int _dev=0):
      BASE(cnine::Gdims(n,nc),fcode,_dev),
      atoms(n){}
    */

    /*
    static Ptensors0bPack cat(const vector<Ptensors0bPack>& list){
      vector<AtomsPack0> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      return Ptensors0bPack(AtomsPack0::cat(v),cnine::Ltensor<TYPE>::stack(0,list));
    }
    */

  public: // ---- Named parameter constructors ---------------------------------------------------------------


    /*
    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    Ptensors0bPack(const AtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
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
    */

  public: // ----- Spawning ----------------------------------------------------------------------------------


    Ptensors0bPack copy() const{
      return Ptensors0bPack(TENSOR::copy(),atoms);
    }

    Ptensors0bPack copy(const int _dev) const{
      return Ptensors0bPack(TENSOR::copy(_dev),atoms);
    }

    Ptensors0bPack zeros_like() const{
      return Ptensors0bPack(TENSOR::zeros_like(),atoms);
    }

    Ptensors0bPack gaussian_like() const{
      return Ptensors0bPack(BASE::gaussian_like(),atoms);
    }

    static Ptensors0bPack zeros_like(const Ptensors0bPack& x){
      return Ptensors0bPack(x.TENSOR::zeros_like(),x.atoms);
    }

    static Ptensors0bPack gaussian_like(const Ptensors0bPack& x){
      return Ptensors0bPack(x.TENSOR::gaussian_like(),x.atoms);
    }

    static Ptensors0bPack* new_zeros_like(const Ptensors0bPack& x){
      return new Ptensors0bPack(x.TENSOR::zeros_like(),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    //Ptensors0bPack(const Ptensors0& x):
    //BASE(cnine::Gdims({x.tail/x.nc,x.nc})),
    //atoms(x.atoms){
    //TENSOR::view2().set(x.view_as_matrix().view2());
    //}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors0bPack(const Ptensors0bPack& x, const int _dev):
      BASE(x.copy(_dev)), 
      atoms(x.atoms){}


  public: // ----- Virtual functions --------------------------------------------------------------------------


    Ptensors0bPack& get_grad(){
      return cnine::diff_class<Ptensors0bPack<TYPE> >::get_grad();
    }

    const Ptensors0bPack& get_grad() const{
      return cnine::diff_class<Ptensors0bPack<TYPE> >::get_grad();
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 0;
    }

    //int size() const{
    //return atoms.size();
    //}

    int get_nc() const{
      return TENSOR::dim(1);
    }

    int nchannels() const{
      return TENSOR::dim(1);
    }

    //AtomsPack get_atoms() const{
    //return atoms.obj->atoms;
    //}

    //int offset(const int i) const{
    //return i;
    //}

    //Atoms atoms_of(const int i) const{
    //return atoms(i);
    //}
    
    //TENSOR tensor_of(const int i) const{
    //return TENSOR::row(offset(i));
    //}

    //Ptensor0 operator()(const int i) const{
    //return Ptensor0(cnine::RtensorA(tensor_of(i).view1()),atoms_of(i));
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Message passing ----------------------------------------------------------------------------


    //template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    //static Ptensors0bPack<float> linmaps(const SOURCE& x){
    //Ptensors0bPack<float> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    //R.add_linmaps(x);
    //return R;
    //}

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0bPack<TYPE> gather(const SOURCE& x, const AtomsPackPack& a){
      int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
      Ptensors0bPack<TYPE> R(a,nc,x.get_dev());
      R.add_gather(x);
      return R;
    }


    template<typename SOURCE>
    void add_gather(const SOURCE& x){
      (atoms.overlaps_mmap(x.atoms))(*this,x);
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      x.atoms.overlaps_mmap(atoms).inv()(*this,x);
    }

    //template<typename OUTPUT>
    //void gather_backprop(const OUTPUT& x){
    //x.atoms.overlaps_mmap(atoms).inv()(get_grad(),x.get_grad());
    //}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors0bPack";
    }

    string repr() const{
      return "<Ptensors0bPack[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	Ptensors0bPack y(*this,0);
	return y.str();
      }
      ostringstream oss;
      //for(int i=0; i<size(); i++){
      //oss<<indent<<(*this)(i)<<endl;
      //}
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors0bPack& x){
      stream<<x.str(); return stream;}


  };




}


#endif 

