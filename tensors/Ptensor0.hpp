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
#ifndef _ptens_Ptensor0
#define _ptens_Ptensor0

#include "Ptens_base.hpp"
#include "Atoms.hpp"
#include "Ltensor.hpp"


namespace ptens{

  template<class TYPE=float>
  class Ptensor0: public cnine::Ltensor<TYPE>{
  public:


    typedef cnine::Gdims Gdims;
    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Rtensor1_view Rtensor1_view;

    using BASE::arr;
    using BASE::dev;
    using BASE::dims;
    using BASE::strides;
    using BASE::view1;

    int k;
    int nc;
    Atoms atoms;

    #ifdef WITH_FAKE_GRAD
    Ptensor0* grad=nullptr;
    #endif 


    ~Ptensor0(){
      #ifdef WITH_FAKE_GRAD
      //if(!is_view) delete grad;
      delete grad;
      #endif 
    }


    // ---- Constructors -------------------------------------------------------------------------------------


    Ptensor0(const Atoms& _atoms, const int _nc, const int _fill, const int _dev=0):
      BASE(cnine::Gdims(_nc),_fill,_dev),
      atoms(_atoms),
      k(_atoms.size()), 
      nc(_nc){
    }

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor0(const Atoms& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      BASE(cnine::Gdims(_nc),dummy,_dev),
      atoms(_atoms),
      k(_atoms.size()), 
      nc(_nc){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor0 raw(const Atoms& _atoms, const int nc=1, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor0 zero(const Atoms& _atoms, const int nc=1, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor0 gaussian(const Atoms& _atoms, const int nc=1, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor0 gaussian(const Atoms& _atoms, const int nc, const float sigma, const int _dev){
      return Ptensor0(_atoms,nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensor0 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Copying ------------------------------------------------------------------------------------------


    Ptensor0(const Ptensor0& x):
      BASE(x.copy()), atoms(x.atoms){
      k=x.k;
      nc=x.nc;
    }

    Ptensor0(Ptensor0&& x):
      BASE(x), atoms(std::move(x.atoms)){
      k=x.k;
      nc=x.nc;
    }

    Ptensor0& operator=(const Ptensor0& x)=delete;


    // ---- Conversions --------------------------------------------------------------------------------------


    Ptensor0(const Atoms& _atoms, const BASE& x):
      BASE(x),
      atoms(_atoms){
      assert(x.ndims()==1);
      k=dims(0);
      nc=dims.back();
    }

    Ptensor0(BASE&& x, Atoms&& _atoms):
      BASE(x),
      atoms(std::move(_atoms)){
      assert(x.ndims()==1);
      k=dims(0);
      nc=dims.back();
    }

    Ptensor0(const Rtensor1_view& x, Atoms&& _atoms):
      BASE(cnine::dims(x.n0),0,x.dev),
      atoms(std::move(_atoms)){
      view().add(x);
      k=dims(0);
      nc=dims.back();
    }

#ifdef _WITH_ATEN
    static Ptensor0 view(at::Tensor& x, Atoms&& _atoms){
      // Check dimensions of x here!
      return Ptensor0(BASE::view(x),std::move(_atoms));
    }
#endif 


    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return dims(0);
    }

    int get_nc() const{
      return dims.back();
    }

    vector<int> atomsv() const{
      return atoms;
    }

    TYPE at_(const int i, const int c) const{
      return (*this)(atoms(i),c);
    }

    void inc_(const int i, const int c, TYPE x){
      inc(atoms(i),c,x);
    }


    Rtensor1_view view() const{
      return view1();
    }

    Rtensor1_view view(const int offs, const int n) const{
      assert(offs+n<=nc);
      return view1().block(offs,n);
    }


  public: // ---- Linmaps -------------------------------------------------------------------------------------


    // 0 <- 0
    void add_linmaps(const Ptensor0& x, int offs=0){ // 1 
      assert(offs+1*x.nc<=nc);
      offs+=broadcast0(x,offs); // 1*1
    }

    void add_linmaps_back(const Ptensor0& x, int offs=0){ // 1 
      assert(offs+1*nc<=x.nc);
      broadcast0(x.reduce0(offs,nc));
    }
    

    // 0 <- 1
    void add_linmaps(const Ptensor1<TYPE>& x, int offs=0) const{ // 1 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+1*nc<=x.nc);
      offs+=broadcast0(x.reduce0(),offs); // 1*1
    }
    
    //void add_linmaps_back(const Ptensor0<TYPE>& , int offs=0){ // 1 
    //PTENS_K_SAME(x)
    //PTENS_CHANNELS(offs+1*nc<=x.nc);
    //view()+=repeat0(x.view(offs,nc),k);
    //}


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      auto R=BASE::zero(Gdims(nc));
      R.view1().add(view());
      return R;
    }

    BASE reduce0(const int offs, const int n) const{
      auto R=BASE::zero(Gdims(n));
      R.view1().add(view(offs,n));
      return R;
    }

    BASE reduce0(const vector<int>& ix) const{
      assert(ix.size()==1);
      auto R=BASE::zero(Gdims(nc));
      R.view1()+=view();
      return R;
    }

    BASE reduce0(const vector<int>& ix, const int offs, const int n) const{
      assert(ix.size()==1);
      auto R=BASE::zero(Gdims(n));
      R.view1()+=view(offs,n);
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& x){
      view(0,x.dim(0))+=x.view1();
    }

    int broadcast0(const BASE& x, const int offs){
      view(offs,x.dim(0))+=x.view1();
      return x.dim(0);
    }

    void broadcast0(const BASE& x, const vector<int>& ix){
      assert(ix.size()==1);
      view(0,x.dim(0))+=x.view1();
    }

    void broadcast0(const BASE& x, const vector<int>& ix, const int offs){
      assert(ix.size()==1);
      view(offs,x.dim(0))+=x.view1();
    }




  private: // ---- Broadcasting -------------------------------------------------------------------------------
    // These methods are deprectated / on hold 

    void broadcast0(const Rtensor1_view& x){
      view()+=x;
    }

    int broadcast0(const Rtensor1_view& x, const int offs){
      view(offs,x.n0)+=x;
      return x.n0;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="")const{
      ostringstream oss;
      oss<<cnine::base_indent<<indent<<"Ptensor0 "<<atoms<<":"<<endl;
      oss<<view1().str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor0& x){
      stream<<x.str(); return stream;}

  };

}


#endif 

