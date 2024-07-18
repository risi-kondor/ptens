/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _ptens_BatchedPtensors2
#define _ptens_BatchedPtensors2

#include "diff_class.hpp"
#include "object_pack.hpp"

#include "BatchedAtomsPack.hpp"
#include "Ptensors2.hpp"
#include "BatchedPtensors.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors2Batch;
  template<typename TYPE> class Ptensors2Batch;


  template<typename TYPE>
  class BatchedPtensors2: public BatchedPtensors<TYPE>,
			   public cnine::diff_class<BatchedPtensors2<TYPE> >{
  public:

    typedef BatchedPtensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    
    using cnine::diff_class<BatchedPtensors2<TYPE> >::grad;
    using BASE::get_dev;

    using TENSOR::dim;

    BatchedAtomsPack atoms;


    ~BatchedPtensors2(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    BatchedPtensors2(const BatchedAtomsPack& _atoms, const TENSOR& M):
      BASE(M), atoms(_atoms){}

    BatchedPtensors2(const TENSOR& M, const BatchedAtomsPack& _atoms):
      BASE(M), atoms(_atoms){}

    BatchedPtensors2(const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.nrows2(),_nc},fcode,_dev), atoms(_atoms){}

    BatchedPtensors2(const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors2(_atoms,_nc,0,_dev){}


    /*
    BatchedPtensors2(const initializer_list<Ptensors2<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPack2obj<int> > > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPackN<AtomsPack2obj<int> >(x);
    }
    */


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    BatchedPtensors2(const BatchedAtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
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


    BatchedPtensors2 copy() const{
      return BatchedPtensors2(atoms,TENSOR::copy());
    }

    BatchedPtensors2 copy(const int _dev) const{
      return BatchedPtensors2(atoms,TENSOR::copy(_dev));
    }

    BatchedPtensors2 zeros_like() const{
      return BatchedPtensors2(atoms,TENSOR::zeros_like());
    }

    BatchedPtensors2 gaussian_like() const{
      return BatchedPtensors2(atoms,TENSOR::gaussian_like());
    }

    static BatchedPtensors2 zeros_like(const BatchedPtensors2& x){
      return BatchedPtensors2(x.atoms,x.TENSOR::zeros_like());
    }

    static BatchedPtensors2 zeros_like(const BatchedPtensors2& x, const int nc){
      return BatchedPtensors2(x.atoms,TENSOR({x.dim(0),nc},0,get_dev()));
    }

    static BatchedPtensors2 gaussian_like(const BatchedPtensors2& x){
      return BatchedPtensors2(x.atoms,x.TENSOR::gaussian_like());
    }

    static BatchedPtensors2* new_zeros_like(const BatchedPtensors2& x){
      return new BatchedPtensors2(x.atoms,x.TENSOR::zeros_like());
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 2;
    }

    int size() const{
      return atoms.size();
    }

    int get_nc() const{
      return TENSOR::dim(1);
    }

    BatchedPtensors2& get_grad(){
      return cnine::diff_class<BatchedPtensors2<TYPE> >::get_grad();
    }

    const BatchedPtensors2& get_grad() const{
      return cnine::diff_class<BatchedPtensors2<TYPE> >::get_grad();
    }

    Ptensors2<TYPE> view_of(const int i) const{
      return Ptensors2<TYPE>(atoms[i],TENSOR::rows(atoms.offset2(i),atoms.nrows2(i)));
    }

    Ptensors2<TYPE> operator[](const int i){
      return Ptensors2<TYPE>(atoms[i],TENSOR::rows(atoms.offset2(i)),atoms.nrows2(i));
    }

    Ptensors2<TYPE> operator[](const int i) const{
      return Ptensors2<TYPE>(atoms[i],TENSOR::rows(atoms.offset2(i),atoms.nrows2(i)));
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors2<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors2<TYPE> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps(x.view_of(i));
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps_back(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps_back(x.view_of(i));
    }

  public: // ---- Gather -------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors2<TYPE> gather(const SOURCE& x, const BatchedAtomsPack& a){
      BatchedPtensors2<TYPE> R(a,x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_gather(x);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x,const int min_overlaps=1){
      add_gather(x,BatchedPtensorMap::overlaps(atoms,x.atoms));
    }

    template<typename SOURCE>
    void add_gather_back(const SOURCE& x,const int min_overlaps=1){
      add_gather_back(x,BatchedPtensorMap::overlaps(x.atoms,atoms));
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const BatchedPtensorMap& map){
      int nc=x.get_nc();
      if constexpr(std::is_same<SOURCE,BatchedPtensors0<TYPE> >::value)
	broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
      if constexpr(std::is_same<SOURCE,BatchedPtensors1<TYPE> >::value){
	broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),2*nc);
      }
	if constexpr(std::is_same<SOURCE,BatchedPtensors2<TYPE> >::value){
	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),4*nc);
	  broadcast2(x.reduce2(map.atoms(),map.in()),map.out(),13*nc);
	}
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const BatchedPtensorMap& map){
      int nc=get_nc();
      if constexpr(std::is_same<OUTPUT,BatchedPtensors0<TYPE> >::value)
	broadcast0_shrink(x.reduce0(map.atoms(),map.out()),map.in());
      if constexpr(std::is_same<OUTPUT,BatchedPtensors1<TYPE> >::value){
	broadcast0_shrink(x.reduce0(map.atoms(),map.out(),0,2*nc),map.in());
	broadcast1_shrink(x.reduce1(map.atoms(),map.out(),2*nc,3*nc),map.in());
      }
      if constexpr(std::is_same<OUTPUT,BatchedPtensors2<TYPE> >::value){
	broadcast0_shrink(x.reduce0_shrink(map.atoms(),map.out(),0,2*nc),map.in());
	broadcast1_shrink(x.reduce1_shrink(map.atoms(),map.out(),4*nc,3*nc),map.in());
	broadcast2(x.reduce2_shrink(map.atoms(),map.out(),13*nc,nc),map.in());
      }
    }

    
  public: // ---- Indexed Reductions -------------------------------------------------------------------------


    BatchedPtensors0<TYPE> reduce0(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc();
      PTENS_ASSRT(offs==0);
      PTENS_ASSRT(nc==get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0<TYPE> R(_atoms,2*nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce0_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors0<TYPE> reduce0_shrink(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list, const int offs, int nc) const{
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0<TYPE> R(_atoms,2*nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce0_shrink_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors1<TYPE> reduce1(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc();
      PTENS_ASSRT(offs==0);
      PTENS_ASSRT(nc==get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors1<TYPE> R(_atoms,3*nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce1_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors1<TYPE> reduce1_shrink(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list, const int offs, int nc) const{
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors1<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce1_shrink_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors2<TYPE> reduce2(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc();
      PTENS_ASSRT(offs==0);
      PTENS_ASSRT(nc==get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors2<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	  view_of(i).add_reduce2_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors2<TYPE> reduce2_shrink(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list, const int offs, int nc) const{
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors2<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce2_shrink_to(R.view_of(i),list[i],offs);
      return R;
    }


 public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const BatchedPtensors0<TYPE>& x, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),offs);
    }

    void broadcast1(const BatchedPtensors1<TYPE>& x, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast1(x.view_of(i),offs);
    }

    void broadcast2(const BASE& x, const int offs=0){
      int nc=x.dim(1);
      BASE::view2().block(0,offs,dim(0),nc)+=x.view2();
    }

    void broadcast0(const BatchedPtensors0<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),list[i],offs);
    }

    void broadcast0_shrink(const BatchedPtensors0<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0_shrink(x.view_of(i),list[i],offs);
    }

    void broadcast1(const BatchedPtensors1<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast1(x.view_of(i),list[i],offs);
    }

    void broadcast1_shrink(const BatchedPtensors1<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast1_shrink(x.view_of(i),list[i],offs);
    }

    void broadcast2(const BatchedPtensors2<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast2(x.view_of(i),list[i],offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedPtensors2";
    }

    string repr() const{
      return "<BatchedPtensors2[N="+to_string(size())+",nrows="+to_string(TENSOR::dim(0))+",nc="+to_string(get_nc())+"]>";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++)
	oss<<(*this)[i]<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors2& x){
      stream<<x.str(); return stream;}


  };

}

#endif 


//     template<typename OUTPUT>
//     void add_gather_back_alt(const OUTPUT& x){
//       int N=size();
//       PTENS_ASSRT(N==x.size());
//       x.backward_program(get_grad(),x.get_grad());
//     }

