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
//#include "BatchedPtensors.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors2Batch;
  template<typename TYPE> class Ptensors2Batch;


  template<typename TYPE>
  class BatchedPtensors2: public BatchedPtensors<TYPE>,
			   public cnine::diff_class<BatchedPtensors2<TYPE> >{
  public:

    typedef BatchedPtensors<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    
    using cnine::diff_class<BatchedPtensors2<TYPE> >::grad;
    using BASE::get_dev;

    using TENSOR::dim;

    BatchedAtomsPack<2> atoms;


    ~BatchedPtensors2(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    BatchedPtensors2(const BatchedAtomsPack<2>& _atoms, const TENSOR& M):
      BASE(M), atoms(_atoms){}

    BatchedPtensors2(const TENSOR& M, const BatchedAtomsPack<2>& _atoms):
      BASE(M), atoms(_atoms){}

    BatchedPtensors2(const BatchedAtomsPack<2>& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.nrows2(),_nc},fcode,_dev), atoms(_atoms){}

    BatchedPtensors2(const BatchedAtomsPack<2>& _atoms, const int _nc, const int _dev):
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
    BatchedPtensors2(const BatchedAtomsPack<2>& _atoms, const Args&... args):
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
    static BatchedPtensors2<TYPE> gather(const BatchedAtomsPack<2>& a, const SOURCE& x){
      BatchedPtensors2<TYPE> R(a,x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_gather(x,BatchedLayerMap::overlaps_map(a,x.atoms));
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors2<TYPE> gather(const BatchedAtomsPack<2>& a, const SOURCE& x, const BatchedLayerMap& map){
      BatchedPtensors2<TYPE> R(a,x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_gather(x,map);
      return R;
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const BatchedLayerMap& map){
      int nc=x.get_nc();
      if constexpr(std::is_same<SOURCE,BatchedPtensors0<TYPE> >::value){
	auto plan0=BatchedGatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	broadcast0(x.reduce0(plan0.in()),plan0.out(),0);
      }
      if constexpr(std::is_same<SOURCE,BatchedPtensors1<TYPE> >::value){
	auto plan0=BatchedGatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	auto plan1=BatchedGatherPlanFactory::gather_map1(map,atoms,x.atoms,2,x.getk());
	broadcast0(x.reduce0(plan0.in()),plan0.out(),0);
	broadcast1(x.reduce1(plan1.in()),plan1.out(),2*nc);
      }
      if constexpr(std::is_same<SOURCE,BatchedPtensors2<TYPE> >::value){
	auto plan0=BatchedGatherPlanFactory::gather_map0(map,atoms,x.atoms,2,x.getk());
	auto plan1=BatchedGatherPlanFactory::gather_map1(map,atoms,x.atoms,2,x.getk());
	auto plan2=BatchedGatherPlanFactory::gather_map2(map,atoms,x.atoms,2,x.getk());
	broadcast0(x.reduce0(plan0.in()),plan0.out(),0);
	broadcast1(x.reduce1(plan1.in()),plan1.out(),4*nc);
	broadcast2(x.reduce2(plan2.in()),plan2.out(),13*nc);
      }
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const BatchedLayerMap& map){
      int nc=get_nc();
      if constexpr(std::is_same<OUTPUT,BatchedPtensors0<TYPE> >::value){
	auto plan0=BatchedGatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),2);
	broadcast0_shrink(x.reduce0(plan0.out()),plan0.in());
      }
      if constexpr(std::is_same<OUTPUT,BatchedPtensors1<TYPE> >::value){
	auto plan0=BatchedGatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),2);
	auto plan1=BatchedGatherPlanFactory::gather_map1(map,x.atoms,atoms,x.getk(),2);
	broadcast0_shrink(x.reduce0(plan0.out(),0,2*nc),plan0.in());
	broadcast1_shrink(x.reduce1(plan1.out(),2*nc,3*nc),plan1.in());
      }
      if constexpr(std::is_same<OUTPUT,BatchedPtensors2<TYPE> >::value){
	auto plan0=BatchedGatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),2);
	auto plan1=BatchedGatherPlanFactory::gather_map1(map,x.atoms,atoms,x.getk(),2);
	auto plan2=BatchedGatherPlanFactory::gather_map2(map,x.atoms,atoms,x.getk(),2);
	broadcast0_shrink(x.reduce0_shrink(plan0.out(),0,2*nc),plan0.in());
	broadcast1_shrink(x.reduce1_shrink(plan1.out(),4*nc,3*nc),plan1.in());
	broadcast2(x.reduce2_shrink(plan2.out(),13*nc,nc),plan2.in());
      }
    }


    
  public: // ---- Indexed Reductions -------------------------------------------------------------------------


    TENSOR reduce0(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors2","reduce0",*this,map,map.size()*get_nc());
      if(nc==0) nc=2*get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce0(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }
    
    TENSOR reduce0_shrink(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors2","reduce0_shrink",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc()/2;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce0_shrink(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }
    
    TENSOR reduce1(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors2","reduce1",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,3*nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce1(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }
    
    TENSOR reduce1_shrink(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors2","reduce1",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc()/3;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce1_shrink(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }
    
    TENSOR reduce2(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors2","reduce2",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce2(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }    

    TENSOR reduce2_shrink(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors2","reduce2_shrink",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc()/2;
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce2_shrink(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
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

    void broadcast0(const TENSOR& x, const BatchedAindexPackB& map, const int offs=0){
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).broadcast0(x.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
    }

    void broadcast0_shrink(const TENSOR& x, const BatchedAindexPackB& map, const int offs=0){
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).broadcast0_shrink(x.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
    }

    void broadcast1(const TENSOR& x, const BatchedAindexPackB& map, const int offs=0){
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).broadcast1(x.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
    }

    void broadcast1_shrink(const TENSOR& x, const BatchedAindexPackB& map, const int offs=0){
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).broadcast1_shrink(x.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
    }

    void broadcast2(const TENSOR& x, const BatchedAindexPackB& map, const int offs=0){
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).broadcast2(x.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
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

//     template<typename SOURCE>
//       void add_gather(const SOURCE& x, const BatchedPtensorMap& map){
//       int nc=x.get_nc();
//       if constexpr(std::is_same<SOURCE,BatchedPtensors0<TYPE> >::value)
// 	broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
//       if constexpr(std::is_same<SOURCE,BatchedPtensors1<TYPE> >::value){
// 	broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
// 	broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),2*nc);
//       }
// 	if constexpr(std::is_same<SOURCE,BatchedPtensors2<TYPE> >::value){
// 	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
// 	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),4*nc);
// 	  broadcast2(x.reduce2(map.atoms(),map.in()),map.out(),13*nc);
// 	}
//     }

//     template<typename OUTPUT>
//     void add_gather_back(const OUTPUT& x, const BatchedPtensorMap& map){
//       int nc=get_nc();
//       if constexpr(std::is_same<OUTPUT,BatchedPtensors0<TYPE> >::value)
// 	broadcast0_shrink(x.reduce0(map.atoms(),map.out()),map.in());
//       if constexpr(std::is_same<OUTPUT,BatchedPtensors1<TYPE> >::value){
// 	broadcast0_shrink(x.reduce0(map.atoms(),map.out(),0,2*nc),map.in());
// 	broadcast1_shrink(x.reduce1(map.atoms(),map.out(),2*nc,3*nc),map.in());
//       }
//       if constexpr(std::is_same<OUTPUT,BatchedPtensors2<TYPE> >::value){
// 	broadcast0_shrink(x.reduce0_shrink(map.atoms(),map.out(),0,2*nc),map.in());
// 	broadcast1_shrink(x.reduce1_shrink(map.atoms(),map.out(),4*nc,3*nc),map.in());
// 	broadcast2(x.reduce2_shrink(map.atoms(),map.out(),13*nc,nc),map.in());
//       }
//     }
