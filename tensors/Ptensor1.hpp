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
#ifndef _ptens_Ptensor1
#define _ptens_Ptensor1

#include "Ptens_base.hpp"
#include "Atoms.hpp"
//#include "RtensorA.hpp"
#include "Ltensor.hpp"
#include "Ptensor0.hpp"
//#include "Ptensor1_xview.hpp"


namespace ptens{

  template<typename TYPE=float>
  class Ptensor1: public cnine::Ltensor<TYPE>{
  public:

    int k;
    int nc;
    Atoms atoms;

    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Gdims Gdims;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;

    using BASE::arr;
    using BASE::dev;
    using BASE::dims;
    using BASE::strides;
    using BASE::view2;


    // ---- Constructors -------------------------------------------------------------------------------------


    Ptensor1(const Atoms& _atoms, const int _nc, const int _fcode, const int _dev=0):
      BASE(cnine::dims(_atoms.size(),_nc),_fcode,_dev),
      atoms(_atoms),
      k(_atoms.size()), 
      nc(_nc){
    }

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor1(const Atoms& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      BASE(cnine::dims(_atoms.size(),_nc),dummy,_dev),
      atoms(_atoms),
      k(_atoms.size()), 
      nc(_nc){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor1 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor1 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor1 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor1 gaussian(const Atoms& _atoms, const int nc, const float sigma, const int _dev){
      return Ptensor1(_atoms,nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensor1 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Copying ------------------------------------------------------------------------------------------


    Ptensor1(const Ptensor1& x):
      BASE(x.copy()), atoms(x.atoms){
      k=x.k;
      nc=x.nc;
    }

    Ptensor1(Ptensor1&& x):
      BASE(x), atoms(std::move(x.atoms)){
      k=x.k;
      nc=x.nc;
    }

    Ptensor1& operator=(const Ptensor1& x)=delete;


    // ---- Conversions --------------------------------------------------------------------------------------


    Ptensor1(const Atoms& _atoms, const BASE& x):
      BASE(x.copy()),
      atoms(_atoms){
      assert(x.ndims()==2);
      k=dims(0);
      nc=dims.back();
     }

    /*
    Ptensor1(const BASE& x, const Atoms& _atoms):
      BASE(x.copy()),
      atoms(_atoms){
      assert(x.ndims()==2);
      k=dims(0);
      nc=dims.back();
     }
    */

    Ptensor1(const cnine::TensorView<TYPE>& x, const Atoms& _atoms):
      BASE(x.copy()),
      atoms(_atoms){
      assert(x.ndims()==2);
      k=dims(0);
      nc=dims.back();
     }

    /*
    Ptensor1(BASE&& x, Atoms&& _atoms):
      BASE(x),
      atoms(std::move(_atoms)){
      assert(x.ndims()==2);
      k=dims(0);
      nc=dims.back();
     }
    */

    #ifdef _WITH_ATEN
    static Ptensor1 view(at::Tensor& x, Atoms&& _atoms){
      // Check dimensions of x here!
      return Ptensor1(BASE::view(x),std::move(_atoms));
    }
    #endif 
 

    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return dims(0);
    }

    int get_nc() const{
      return dims.back();
    }

    TYPE at_(const int i, const int c) const{
      return (*this)(atoms(i),c);
    }

    void inc_(const int i, const int c, TYPE x){
      inc(atoms(i),c,x);
    }


    Rtensor2_view view() const{
      return view2();
    }

    Rtensor2_view view(const int offs, const int n) const{
      assert(offs+n<=nc);
      return view2().block(0,offs,k,n);
    }

    //Ptensor1_xview view(const vector<int>& ix) const{
    //return Ptensor1_xview(const_cast<TYPE*>(arr.get_arr()),nc,strides[0],strides[1],ix,dev);
    //}

    //Ptensor1_xview view(const vector<int>& ix, const int offs, const int n) const{
    //return Ptensor1_xview(const_cast<TYPE*>(arr.get_arr())+strides[1]*offs,n,strides[0],strides[1],ix,dev);
    //}


    // ---- Linmaps ------------------------------------------------------------------------------------------


    // 1 <- 0
    void add_linmaps(const Ptensor0<TYPE>& x, int offs=0){ // 1
      PTENS_K_SAME(x);
      PTENS_CHANNELS(offs+1*x.nc<=nc);
      offs+=broadcast0(x.view1(),offs); // 1*1
    }
    
    void add_linmaps_back_to(Ptensor0<TYPE>& x, int offs=0) const{ // 1
      PTENS_K_SAME(x)
      view(offs,x.nc).sum0_into(x.view1());
    }

    
    // 1 <- 1 
    void add_linmaps(const Ptensor1& x, int offs=0){ // 2 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+2*x.nc<=nc);
      offs+=broadcast0(x.reduce0().view1(),offs); // 1*1
      offs+=broadcast1(x.view(),offs); // 1*1
    }
    
    void add_linmaps_back(const Ptensor1& x, int offs=0){ // 2 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+2*nc<=x.nc);
      broadcast0(x.reduce0(offs,nc).view1()); // corrected!!
      broadcast1(x.view(offs+nc,nc));
    }
    

    // 1 -> 0
    /*
    void add_linmaps_to(Ptensor0<TYPE>& x, int offs=0) const{ // 1 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+1*nc<=x.nc);
      offs+=x.broadcast0(reduce0(),offs); // 1*1
    }
    
    void add_linmaps_back(const Ptensor0<TYPE>& x, int offs=0){ // 1 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+1*nc<=x.nc);
      view()+=repeat0(x.view(offs,nc),k);
    }
    */


  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{
      auto R=BASE::zero(nc);
      view().sum0_into(R.view1());
      return R;
    }

    BASE reduce0(const int offs, const int n) const{
      auto R=BASE::zero(n);
      view(offs,n).sum0_into(R.view1());
      return R;
    }

    BASE reduce0(const vector<int>& ix) const{
      auto R=BASE::zero(Gdims(nc));
      view(ix).sum0_into(R.view1());
      return R;
    }

    BASE reduce0(const vector<int>& ix, const int offs, const int n) const{
      auto R=BASE::zero(Gdims(n));
      view(ix,offs,n).sum0_into(R.view1());
      return R;
    }


    BASE reduce1() const{
      auto R=BASE::zero({k,nc});
      R.view2().add(view());
      return R;
    }

    BASE reduce1(const int offs, const int n) const{
      auto R=BASE::zero({k,n});
      R.view2().add(view(offs,n));
      return R;
    }

    BASE reduce1(const vector<int>& ix) const{
      auto R=BASE::zero({(int)ix.size(),nc});
      R.view2().add(view(ix));
      return R;
    }

    BASE reduce1(const vector<int>& ix, const int offs, const int n) const{
      auto R=BASE::zero({(int)ix.size(),n});
      R.view2().add(view(ix,offs,n));
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BASE& x){
      view()+=repeat0(x.view1(),k);
    }

    int broadcast0(const BASE& x, const int offs){
      view(offs,x.dim(0))+=repeat0(x.view1(),k);
      return x.dim(0);
    }

    void broadcast0(const BASE& x, const vector<int>& ix){
      view(ix)+=repeat0(x.view1(),ix.size());
    }

    int broadcast0(const BASE& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.dim(0))+=repeat0(x.view1(),ix.size());
      return x.dim(0);
    }


    void broadcast1(const BASE& x){
      view2().add(x.view2());
    }

    int broadcast1(const BASE& x, const int offs){
      view(offs,x.dim(1))+=x.view2();
      return x.dim(1);
    }

    void broadcast1(const BASE& x, const vector<int>& ix){
      view(ix)+=x.view2();
    }

    int broadcast1(const BASE& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.dim(1))+=x.view2();
      return x.dim(1);
    }


  private: // ---- Broadcasting -------------------------------------------------------------------------------
    // These methods are deprectated / on hold 


    void broadcast0(const Rtensor1_view& x){
      view()+=repeat0(x,k);
    }

    int broadcast0(const Rtensor1_view& x, const int offs){
      view(offs,x.n0)+=repeat0(x,k);
      return x.n0;
    }

    void broadcast0(const Rtensor1_view& x, const vector<int>& ix){
      view(ix)+=repeat0(x,ix.size());
    }

    int broadcast0(const Rtensor1_view& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.n0)+=repeat0(x,ix.size());
      return x.n0;
    }


    void broadcast1(const Rtensor2_view& x){
      view2().add(x);
    }

    int broadcast1(const Rtensor2_view& x, const int offs){
      view(offs,x.n1)+=x;
      return x.n1;
    }

    void broadcast1(const Rtensor2_view& x, const vector<int>& ix){
      view(ix)+=x;
    }

    int broadcast1(const Rtensor2_view& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.n1)+=x;
      return x.n1;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="")const{
      ostringstream oss;
      oss<<cnine::base_indent<<indent<<"Ptensor1 "<<atoms<<":"<<endl;
      oss<<view2().str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor1& x){
      stream<<x.str(); return stream;}

  };

}


#endif 

