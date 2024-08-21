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

#ifndef _ptens_Ptensor2
#define _ptens_Ptensor2

#include "Ptens_base.hpp"
#include "Atoms.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2_xview.hpp"

namespace ptens{

  template<typename TYPE=float>
  class Ptensor2: public cnine::Ltensor<TYPE>{
  public:

    int k;
    int nc;
    Atoms atoms;

    typedef cnine::Gdims Gdims;
    typedef cnine::Ltensor<TYPE> BASE;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    using BASE::arr;
    using BASE::dev;
    using BASE::dims;
    using BASE::strides;
    using BASE::view3;


    // ---- Constructors -------------------------------------------------------------------------------------


    Ptensor2(const Atoms& _atoms, const int _nc, const int _fcode, const int _dev=0):
      BASE(cnine::dims(_atoms.size(),_atoms.size(),_nc),_fcode,_dev),
      atoms(_atoms), k(_atoms.size()), nc(_nc){
    }

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor2(const Atoms& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      BASE(cnine::dims(_atoms.size(),_atoms.size(),_nc),dummy,_dev),
      atoms(_atoms), k(_atoms.size()), nc(_nc){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor2 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor2 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor2 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor2 gaussian(const Atoms& _atoms, const int nc, const float sigma, const int _dev){
      return Ptensor2(_atoms,nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensor2 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Copying ------------------------------------------------------------------------------------------


    Ptensor2(const Ptensor2& x):
      BASE(x.copy()), atoms(x.atoms){
      k=x.k;
      nc=x.nc;
    }

    Ptensor2(Ptensor2&& x):
      BASE(x), atoms(std::move(x.atoms)){
      k=x.k;
      nc=x.nc;
    }

    Ptensor2& operator=(const Ptensor2& x)=delete;


    // ---- Conversions --------------------------------------------------------------------------------------


    Ptensor2(const Atoms& _atoms, const BASE& x):
      BASE(x),
      atoms(_atoms){
      assert(x.ndims()==3);
      k=dims(0);
      nc=dims.back();
    }
 
    Ptensor2(BASE&& x, Atoms&& _atoms):
      BASE(x),
      atoms(std::move(_atoms)){
      assert(x.ndims()==3);
      k=dims(0);
      nc=dims.back();
    }
 

    Ptensor2(const Rtensor3_view& x, Atoms&& _atoms):
      BASE(cnine::dims(x.n0,x.n1,x.n2),0,x.dev),
      atoms(std::move(_atoms)){
      view().add(x);
      k=dims(0);
      nc=dims.back();
    }

    #ifdef _WITH_ATEN
    //static Ptensor2 view(at::Tensor& x, Atoms&& _atoms){
    //return Ptensor2(BASE::view(x),std::move(_atoms));
    //}
    #endif 


    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return k;
    }

    int get_nc() const{
      return nc;
      //      return dims.back();
    }

    TYPE at_(const int i, const int j, const int c) const{
      return (*this)(atoms(i),atoms(j),c);
    }

    void inc_(const int i, const int j, const int c, TYPE x){
      inc(atoms(i),atoms(j),c,x);
    }


    Rtensor3_view view() const{
      return view3();
    }

    Rtensor3_view view(const int offs, const int n) const{
      assert(offs+n<=nc);
      return view3().block(0,0,offs,k,k,n);
    }
    
    Ptensor2_xview view(const vector<int>& ix) const{
      return Ptensor2_xview(const_cast<TYPE*>(arr.get_arr()),nc,strides[0],strides[1],strides[2],ix,dev);
    }

    Ptensor2_xview view(const vector<int>& ix, const int offs, const int n) const{
      return Ptensor2_xview(const_cast<TYPE*>(arr.get_arr())+strides[2]*offs,n,strides[0],strides[1],strides[2],ix,dev);
    }


    // ---- Linmaps ------------------------------------------------------------------------------------------


    // 0 -> 2
    void add_linmaps(const Ptensor0<TYPE>& x, int offs=0){ // 2
      assert(offs+2*x.nc<=nc);
      offs+=broadcast0(x,offs); // 2*1
    }

    void add_linmaps_back_to(Ptensor0<TYPE>& x, int offs=0) const{ // 2
      assert(offs+2*x.nc<=nc);
      x.add(reduce0(offs,x.nc));
    }


    // 1 -> 2
    void add_linmaps(const Ptensor1<TYPE>& x, int offs=0){ // 5 
      assert(x.k==k);
      assert(offs+5*x.nc<=nc);
      offs+=broadcast0(x.reduce0(),offs); // 2*1
      offs+=broadcast1(x,offs); // 3*1
    }

    void add_linmaps_back_to(Ptensor1<TYPE>& x, int offs=0) const{ // 5 
      assert(x.k==k);
      assert(offs+5*x.nc<=nc);
      x.broadcast0(reduce0(offs,x.nc));
      x.broadcast1(reduce1(offs+2*x.nc,x.nc));
    }
    

    // 2 -> 2
    void add_linmaps(const Ptensor2& x, int offs=0){ // 15
      assert(x.k==k);
      assert(offs+15*x.nc<=nc);
      offs+=broadcast0(x.reduce0(),offs); // 2*2
      offs+=broadcast1(x.reduce1(),offs); // 3*3
      offs+=broadcast2(x,offs); // 2
    }
    
    void add_linmaps_back(const Ptensor2& x, int offs=0){ // 15 check offsets!!!
      assert(x.k==k);
      assert(offs+15*nc<=x.nc);
      broadcast0(x.reduce0(offs,nc)); // 2*2
      broadcast1(x.reduce1(offs+2*nc,nc)); // 3*3
      view3().add(x.view(offs+5*nc,nc)); // 2 
      //broadcast2(x.view(offs+5*nc,nc)); // 2 
    }
    

    // 2 -> 0 
    void add_linmaps_to(Ptensor0<TYPE>& x, int offs=0) const{ // 2
      assert(offs+2*nc<=x.nc);
      offs+=x.broadcast0(reduce0(),offs); // 1*2
    }
    
    void add_linmaps_back(const Ptensor0<TYPE>& x, int offs=0){ // 2
      assert(offs+2*nc<=x.nc);
      //offs+=x.broadcast(reduce0().view1(),offs); // 1*2
    }
    

    // 2 -> 1
    void add_linmaps_to(Ptensor1<TYPE>& x, int offs=0) const{ // 5 
      assert(x.k==k);
      assert(offs+5*nc<=x.nc);
      offs+=x.broadcast0(reduce0(),offs); // 1*2
      offs+=x.broadcast1(reduce1(),offs); // 1*3
    }
    
    void add_linmaps_back(const Ptensor1<TYPE>& x, int offs=0){ // 5 
      assert(x.k==k);
      assert(offs+5*nc<=x.nc);
      //offs+=x.broadcast(reduce0().view1(),offs); // 1*2
      //offs+=x.broadcast(reduce1().view2(),offs); // 1*3
    }
    

  public: // ---- Reductions ---------------------------------------------------------------------------------


    BASE reduce0() const{ // 2
      auto R=BASE::zero({2*nc});
      view().sum01_into(R.view1().block(0,nc));
      view().diag01().sum0_into(R.view1().block(nc,nc));
      return R;
    }

    BASE reduce0(const int offs, const int n) const{ // 2
      auto R=BASE::zero({n});
      view(offs,n).sum01_into(R.view1());
      view(offs+n,n).diag01().sum0_into(R.view1());
      return R;
    }

    BASE reduce0(const vector<int>& ix) const{
      auto R=BASE::zero(Gdims(2*nc));
      view(ix).sum01_into(R.view1().block(0,nc));
      view(ix).diag01().sum0_into(R.view1().block(nc,nc));
      return R;
    }

    BASE reduce0(const vector<int>& ix, const int offs, const int n) const{
      auto R=BASE::zero(Gdims(n));
      view(ix,offs,n).sum01_into(R.view1());
      view(ix,offs+n,n).diag01().sum0_into(R.view1());
      return R;
    }


    BASE reduce1() const{
      auto R=BASE::zero({k,3*nc});
      view().sum0_into(R.view2().block(0,0,k,nc));
      view().sum1_into(R.view2().block(0,nc,k,nc));
      R.view2().block(0,2*nc,k,nc)+=view().diag01();
      return R;
    }

    BASE reduce1(const int offs, const int n) const{
      auto R=BASE::zero({k,n});
      view(offs,n).sum0_into(R.view2());
      view(offs+n,n).sum1_into(R.view2());
      R.view2()+=view(offs+2*n,n).diag01();
      return R;
    }

    BASE reduce1(const vector<int>& ix) const{
      int K=ix.size();
      auto R=BASE::zero({K,3*nc});
      view(ix).sum0_into(R.view2().block(0,0,K,nc));
      view(ix).sum1_into(R.view2().block(0,nc,K,nc));
      R.view2().block(0,2*nc,K,nc).add(view(ix).diag01());
      return R;
    }

    BASE reduce1(const vector<int>& ix, const int offs, const int n) const{
      auto R=BASE::zero({(int)ix.size(),n});
      view(ix,offs,n).sum0_into(R.view2());
      view(ix,offs+n,n).sum1_into(R.view2());
      R.view2().add(view(ix,offs+2*n,n).diag01());
      return R;
    }


    BASE reduce2() const{
      auto R=BASE::zero({k,k,nc});
      R.view3().add(view());
      return R;
    }

    BASE reduce2(const int offs, const int n) const{ // flipping
      auto R=BASE::zero({k,k,n});
      R.view3().block(0,0,0,k,k,n).add(view(offs,n));
      R.view3().block(0,0,n,k,k,n).add(view(offs+n,n).transp01());
      return R;
    }

    BASE reduce2(const vector<int>& ix) const{
      auto R=BASE::zero({(int)ix.size(),(int)ix.size(),nc});
      R.view3().add(view(ix));
      return R;
    }

    BASE reduce2(const vector<int>& ix, const int offs, const int n) const{ // flipping
      int K=ix.size();
      auto R=BASE::zero({(int)ix.size(),(int)ix.size(),n});
      R.view3().block(0,0,0,K,K,n).add(view(ix,offs,n));
      R.view3().block(0,0,0,K,K,n).add(view(ix,offs+n,n).transp01());
      return R;
    }

    
  public: // ---- Broadcasting -------------------------------------------------------------------------------


    int broadcast0(const BASE& x, const int offs){
      int n=x.dim(0);
      assert(2*n+offs<=nc);
      view(offs,n)+=repeat0(repeat0(x.view1(),k),k);
      view(offs+n,n).diag01()+=repeat0(x.view1(),k);
      return 2*n;
    }

    void broadcast0(const BASE& x){
      int n=x.dim(0);
      assert(n==2*nc);
      view()+=repeat0(repeat0(x.view1().block(0,nc),k),k);
      view().diag01()+=repeat0(x.view1().block(nc,nc),k);
    }

    int broadcast0(const BASE& x, const vector<int>& ix, const int offs){
      int n=x.dim(0);
      int K=ix.size();
      assert(2*n+offs<=nc);
      view(ix,offs,n)+=repeat0(repeat0(x.view1(),K),K);
      view(ix,offs+n,n).diag01()+=repeat0(x.view1(),K);
      return 2*n;
    }

    void broadcast0(const BASE& x, const vector<int>& ix){
      int n=x.dim(0);
      int K=ix.size();
      assert(n==2*nc);
      view(ix)+=repeat0(repeat0(x.view1().block(0,nc),K),K);
      view(ix).diag01()+=repeat0(x.view1().block(nc,nc),K);
    }


    int broadcast1(const BASE& x, const int offs){
      int n=x.dim(1);
      assert(3*n+offs<=nc);
      view(offs,n)+=repeat0(x.view2(),k);
      view(offs+n,n)+=repeat1(x.view2(),k);
      view(offs+2*n,n).diag01()+=x.view2();
      return 3*n;
    }

    void broadcast1(const BASE& x){
      int n=x.dim(1);
      assert(n==3*nc);
      view()+=repeat0(x.view2().block(0,0,k,nc),k);
      view()+=repeat1(x.view2().block(0,nc,k,nc),k);
      view().diag01()+=x.view2().block(0,2*nc,k,nc);
    }

    int broadcast1(const BASE& x, const vector<int>& ix, const int offs){
      int n=x.dim(1);
      int K=ix.size();
      assert(3*n+offs<=nc);
      view(ix,offs,n)+=repeat0(x.view2(),K);
      view(ix,offs+n,n)+=repeat1(x.view2(),K);
      view(ix,offs+2*n,n).diag01()+=x.view2();
      return 3*n;
    }

    void broadcast1(const BASE& x, const vector<int>& ix){
      int n=x.dim(1);
      int K=ix.size();
      assert(n==3*nc);
      view(ix)+=repeat0(x.view2().block(0,0,K,nc),K);
      view(ix)+=repeat1(x.view2().block(0,nc,K,nc),K);
      view(ix).diag01()+=x.view2().block(0,2*nc,K,nc);
    }

  
    int broadcast2(const BASE& x, const int offs){
      int n=x.dim(2);
      assert(2*n+offs<=nc);
      view(offs,n)+=x.view3();
      view(offs+n,n)+=x.view3().transp01();
      return 2*n;
    }

    void broadcast2(const BASE& x){ // no flipping
      int n=x.dim(2);
      assert(n==nc);
      view()+=x.view3();
    }

    int broadcast2(const BASE& x, const vector<int>& ix, const int offs){
      int n=x.dim(2);
      assert(2*n+offs<=nc);
      view(ix,offs,n)+=x.view3();
      view(ix,offs+n,n)+=x.view3().transp01();
      return 2*n;
    }

    void broadcast2(const BASE& x, const vector<int>& ix){ // no flipping
      int n=x.dim(2);
      assert(n==nc);
      view(ix)+=x.view3();
    }


  private: // ---- Broadcasting -------------------------------------------------------------------------------
    // These methods are deprectated / on hold 

    /*
    int broadcast(const Rtensor1_view& x, const int offs){
      int n=x.n0;
      assert(2*n+offs<=nc);
      view(offs,n)+=repeat0(repeat0(x,k),k);
      view(offs+n,n).diag01()+=repeat0(x,k);
      return 2*n;
    }

    void broadcast(const Rtensor1_view& x){
      int n=x.n0;
      assert(n==2*nc);
      view()+=repeat0(repeat0(x.block(0,n),k),k);
      view().diag01()+=repeat0(x.block(n,n),k);
    }

    int broadcast(const Rtensor1_view& x, const vector<int>& ix, const int offs){
      int n=x.n0;
      int K=ix.size();
      assert(2*n+offs<=nc);
      view(ix,offs,n)+=repeat0(repeat0(x,K),K);
      view(ix,offs+n,n).diag01()+=repeat0(x,K);
      return 2*n;
    }

    void broadcast(const Rtensor1_view& x, const vector<int>& ix){
      int n=x.n0;
      int K=ix.size();
      assert(n==2*nc);
      view(ix)+=repeat0(repeat0(x.block(0,n),K),K);
      view(ix).diag01()+=repeat0(x.block(0,n),K);
    }


    int broadcast(const Rtensor2_view& x, const int offs){
      int n=x.n1;
      assert(3*n+offs<=nc);
      view(offs,n)+=repeat0(x,k);
      view(offs+n,n)+=repeat1(x,k);
      view(offs+2*n,n).diag01()+=x;
      return 3*n;
    }

    void broadcast(const Rtensor2_view& x){
      int n=x.n1;
      assert(n==3*nc);
      view()+=repeat0(x.block(0,0,k,nc),k);
      view()+=repeat1(x.block(0,nc,k,nc),k);
      view().diag01()+=x.block(0,2*nc,k,nc);
    }

    int broadcast(const Rtensor2_view& x, const vector<int>& ix, const int offs){
      int n=x.n1;
      int K=ix.size();
      assert(3*n+offs<=nc);
      view(ix,offs,n)+=repeat0(x,K);
      view(ix,offs+n,n)+=repeat1(x,K);
      view(ix,offs+2*n,n).diag01()+=x;
      return 3*n;
    }

    void broadcast(const Rtensor2_view& x, const vector<int>& ix){
      int n=x.n1;
      int K=ix.size();
      assert(n==3*nc);
      view(ix)+=repeat0(x.block(0,0,K,nc),K);
      view(ix)+=repeat1(x.block(0,nc,K,nc),K);
      view(ix).diag01()+=x.block(0,2*nc,K,nc);
    }

  
    int broadcast(const Rtensor3_view& x, const int offs){
      int n=x.n2;
      assert(2*n+offs<=nc);
      view(offs,n)+=x;
      view(offs+n,n)+=x.transp01();
      return 2*n;
    }

    void broadcast(const Rtensor3_view& x){ // no flipping
      int n=x.n2;
      assert(n==nc);
      view()+=x;
    }

    int broadcast(const Rtensor3_view& x, const vector<int>& ix, const int offs){
      int n=x.n2;
      assert(2*n+offs<=nc);
      view(ix,offs,n)+=x;
      view(ix,offs+n,n)+=x.transp01();
      return 2*n;
    }

    void broadcast(const Rtensor3_view& x, const vector<int>& ix){ // no flipping
      int n=x.n2;
      assert(n==nc);
      view(ix)+=x;
    }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Ptensor2 "<<atoms<<":"<<endl;
      for(int c=0; c<get_nc(); c++){
	oss<<indent<<"  channel "<<c<<":"<<endl;
	oss<<view3().slice2(c).str(indent+"    ");
	if(c<get_nc()-1) oss<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const Ptensor2& x){
      stream<<x.str(); return stream;}

  };


}


#endif 

