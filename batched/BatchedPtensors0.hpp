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

#ifndef _ptens_BatchedPtensors0
#define _ptens_BatchedPtensors0

#include "diff_class.hpp"
#include "object_pack.hpp"

#include "BatchedAtomsPack.hpp"
#include "BatchedPtensorMap.hpp"
#include "Ptensors0.hpp"
#include "BatchedPtensors.hpp"
#include "MultiLoop.hpp"


namespace ptens{


  template<typename TYPE>
  class BatchedPtensors0: public BatchedPtensors<TYPE>,
			   public cnine::diff_class<BatchedPtensors0<TYPE> >{
  public:

    typedef BatchedPtensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    
    using cnine::diff_class<BatchedPtensors0<TYPE> >::grad;
    using BASE::get_dev;

    BatchedAtomsPack atoms;


    ~BatchedPtensors0(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    BatchedPtensors0(const BatchedAtomsPack& _atoms, const TENSOR& M):
      BASE(M), atoms(_atoms){}

    //BatchedPtensors0(const TENSOR& M, const BatchedAtomsPack& _atoms):
    //BASE(M), atoms(_atoms){}

    BatchedPtensors0(const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors0(_atoms,_nc,0,_dev){}

    BatchedPtensors0(const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.nrows0(),_nc},fcode,_dev), atoms(_atoms){}


    BatchedPtensors0(const initializer_list<Ptensors0<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPackObj> > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPack(BatchedAtomsPackObj(x));
    }
	

  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    BatchedPtensors0(const BatchedAtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.nrows0(),v.nc},v.fcode,v.dev));
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


    BatchedPtensors0 copy() const{
      return BatchedPtensors0(atoms,TENSOR::copy());
    }

    BatchedPtensors0 copy(const int _dev) const{
      return BatchedPtensors0(atoms,TENSOR::copy(_dev));
    }

    BatchedPtensors0 zeros_like() const{
      return BatchedPtensors0(atoms,TENSOR::zeros_like());
    }

    BatchedPtensors0 gaussian_like() const{
      return BatchedPtensors0(atoms,TENSOR::gaussian_like());
    }

    static BatchedPtensors0 zeros_like(const BatchedPtensors0& x){
      return BatchedPtensors0(x.TENSOR::zeros_like(),x.atoms);
    }

    static BatchedPtensors0 zeros_like(const BatchedPtensors0& x, const int nc){
      return BatchedPtensors0(TENSOR({x.dim(0),nc},0,get_dev()),x.atoms);
    }

    static BatchedPtensors0 gaussian_like(const BatchedPtensors0& x){
      return BatchedPtensors0(x.atoms,x.TENSOR::gaussian_like());
    }

    static BatchedPtensors0* new_zeros_like(const BatchedPtensors0& x){
      return new BatchedPtensors0(x.atoms,x.TENSOR::zeros_like());
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    //BatchedPtensors0(const TENSOR& x, const BatchedAtomsPack0& _atoms):
    //BASE(x),
    //atoms(_atoms){}


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

    BatchedPtensors0& get_grad(){
      return cnine::diff_class<BatchedPtensors0<TYPE> >::get_grad();
    }

    const BatchedPtensors0& get_grad() const{
      return cnine::diff_class<BatchedPtensors0<TYPE> >::get_grad();
    }

    Ptensors0<TYPE> view_of(const int i) const{
      return Ptensors0<TYPE>(atoms[i],TENSOR::rows(atoms.offset0(i),atoms.nrows0(i)));
    }

    Ptensors0<TYPE> operator[](const int i){
      return Ptensors0<TYPE>(atoms[i],TENSOR::rows(atoms.offset0(i)),atoms.nrows0(i));
    }

    Ptensors0<TYPE> operator[](const int i) const{
      return Ptensors0<TYPE>(atoms[i],TENSOR::rows(atoms.offset0(i),atoms.nrows0(i)));
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors0<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors0<TYPE> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps(const SOURCE& x){
      cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps(x.view_of(i));});
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps_back(const SOURCE& x){
      cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps_back(x.view_of(i));});
    }


  public: // ---- Gather --------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors0<TYPE> gather(const BatchedAtomsPack& a, const SOURCE& x, const int min_overlaps=1){
      BatchedPtensors0<TYPE> R(a,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_gather(x,min_overlaps);
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
      broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const BatchedPtensorMap& map){
      int nc=get_nc();
      if constexpr(std::is_same<OUTPUT,BatchedPtensors0<TYPE> >::value)
	broadcast0(x.reduce0(map.atoms(),map.out()),map.in());
      if constexpr(std::is_same<OUTPUT,BatchedPtensors1<TYPE> >::value)
	broadcast0(x.reduce0(map.atoms(),map.out()),map.in());
      if constexpr(std::is_same<OUTPUT,BatchedPtensors2<TYPE> >::value)
	broadcast0(x.reduce0_shrink(map.atoms(),map.out(),0,nc),map.in());
    }

    
  public: // ---- Reductions ---------------------------------------------------------------------------------


    TENSOR reduce0() const{
      return *this;
    }

    BatchedPtensors0 reduce0(const BatchedAtomsPack& _atoms, const BatchedAindexPack& list) const{
      TimedFn T("BatchedPtensors0","reduce0",*this,list,list.size()*get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0 R(_atoms,get_nc(),0,get_dev());
      for(int i=0; i<size(); i++){
	R.view_of(i)+=view_of(i).reduce0(_atoms[i],list[i]);
      }
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const TENSOR& x, const int offs=0){
      TENSOR::view2().cols(offs,x.dim(1))+=x.view2();
    }

    void broadcast0(const BatchedPtensors0& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),list[i],offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedPtensors0";
    }

    string repr() const{
      return "<BatchedPtensors0[N="+to_string(size())+",nrows="+to_string(TENSOR::dim(0))+",nc="+to_string(get_nc())+"]>";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++)
	oss<<(*this)[i]<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors0& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

    //int N=size();
    //PTENS_ASSRT(N==x.size());
    //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_gather(x.view_of(i));});
    //for(int i=0; i<N; i++){
      //MessageList mlist=atoms.obj->obj[i]->atoms->overlaps_mlist(*x.atoms.obj->obj[i]->atoms,min_overlaps);
      //MessageMap mmap=atoms.obj->obj[i]->message_map(*mlist.obj,*x.atoms.obj->obj[i]);
      //forward_program.obj.push_back(mmap.obj);
      //backward_program.obj.push_back(to_share(new cnine::GatherMapProgram(mmap.obj->inv()))); // eliminate the copy here 
      //}
      //forward_program(*this,x);
    //template<typename OUTPUT>
    //void add_gather_back(const OUTPUT& x){
    //int N=size();
    //PTENS_ASSRT(N==x.size());
    //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_gather_back(x.view_of(i));});
      //cnine::GatherMapProgramPack P;
      //for(int i=0; i<N; i++){
      //MessageList mlist=x.atoms.obj->obj[i]->atoms->overlaps_mlist(*atoms.obj->obj[i]->atoms);
      //MessageMap mmap=x.atoms.obj->obj[i]->message_map(*mlist.obj,*atoms.obj->obj[i]);
      //P.obj.push_back(to_share(new cnine::GatherMapProgram(mmap.obj->inv()))); // eliminate the copy here 
      //}
      //P(*this,x);
    //}

