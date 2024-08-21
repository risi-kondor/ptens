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

#ifndef _ptens_BatchedPtensors1
#define _ptens_BatchedPtensors1

#include "diff_class.hpp"
#include "object_pack.hpp"

#include "BatchedAtomsPack.hpp"
#include "Ptensors1.hpp"
//#include "BatchedPtensors.hpp"
#include "MultiLoop.hpp"
#include "BatchedGatherPlanFactory.hpp"


namespace ptens{


  template<typename TYPE>
  class BatchedPtensors1: public BatchedPtensors<TYPE>,
			   public cnine::diff_class<BatchedPtensors1<TYPE> >{
  public:

    typedef BatchedPtensors<TYPE> BASE;
    typedef typename BASE::TENSOR TENSOR;
    
    using cnine::diff_class<BatchedPtensors1<TYPE> >::grad;
    using BASE::get_dev;

    using TENSOR::dim;
    using TENSOR::get_arr;

    BatchedAtomsPack<1> atoms;


    ~BatchedPtensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    BatchedPtensors1(const BatchedAtomsPackBase& _atoms, const TENSOR& M):
      BASE(M), atoms(_atoms){}

    BatchedPtensors1(const TENSOR& M, const BatchedAtomsPack<1>& _atoms):
      BASE(M), atoms(_atoms){}

    BatchedPtensors1(const BatchedAtomsPackBase& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.nrows1(),_nc},fcode,_dev), atoms(_atoms){}


    BatchedPtensors1(const BatchedAtomsPackBase& _atoms, const int _nc, const int _dev):
      BatchedPtensors1(_atoms,_nc,0,_dev){}


    //BatchedPtensors1(TYPE* _arr, const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
    //BASE(_arr,{_atoms.nrows1(),_nc},_dev), atoms(_atoms){}


    // TODO 
    BatchedPtensors1(const initializer_list<Ptensors1<TYPE> >& list):
      BASE(PtensTensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPackObj> > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPack<1>(x);
    }
	
    static BatchedPtensors1 cat(const vector<BatchedPtensors1>& list){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      PTENS_ASSRT(list.size()>0);
      vector<BatchedAtomsPackBase> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      BatchedPtensors1 R(BatchedAtomsPackBase::cat(v),list[0].get_nc(),list[0].get_dev());

      int N=list[0].size();
      for(int i=0; i<N; i++){
	vector<PtensTensor<TYPE> > w;
	for(int j=0; j<list.size(); j++)
	  w.push_back(list[j].view_of(i));
	R.view_of(i).cnine:: template PtensTensor<TYPE>::operator=(PtensTensor<TYPE>::stack(0,w));
      }
      return R;
    }


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    BatchedPtensors1(const BatchedAtomsPackBase& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.nrows1(),v.nc},v.fcode,v.dev));
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


    BatchedPtensors1 copy() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(TENSOR::copy(),atoms);
    }

    BatchedPtensors1 copy(const int _dev) const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(TENSOR::copy(_dev),atoms);
    }

    BatchedPtensors1 zeros_like() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(TENSOR::zeros_like(),atoms);
    }

    BatchedPtensors1 zeros_like(const int nc) const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(TENSOR({dim(0),nc},0,get_dev()),atoms);
    }

    BatchedPtensors1 gaussian_like() const{
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(TENSOR::gaussian_like(),atoms);
    }

    static BatchedPtensors1 zeros_like(const BatchedPtensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(x.BASE::zeros_like(),x.atoms);
    }

    static BatchedPtensors1 zeros_like(const BatchedPtensors1& x, const int nc){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(TENSOR({x.dim(0),nc},0,get_dev()),x.atoms);
    }

    static BatchedPtensors1 gaussian_like(const BatchedPtensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1(x.TENSOR::gaussian_like(),x.atoms);
    }

    static BatchedPtensors1* new_zeros_like(const BatchedPtensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new BatchedPtensors1(x.BASE::zeros_like(),x.atoms);
    }
    
    static BatchedPtensors1* new_like(TYPE* _arr, const BatchedPtensors1& x){
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new BatchedPtensors1(x.TENSOR::like(_arr),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------



  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 1;
    }

    int size() const{
      return atoms.size();
    }

    int get_nc() const{
      return TENSOR::dim(1);
    }

    BatchedPtensors1& get_grad(){ // why do we need these?
      return cnine::diff_class<BatchedPtensors1<TYPE> >::get_grad();
    }

    const BatchedPtensors1& get_grad() const{
      return cnine::diff_class<BatchedPtensors1<TYPE> >::get_grad();
    }

    BatchedPtensors1& get_grad(TYPE* _arr){
      if(!grad) grad=new_like(_arr,*this);
      return *grad;
    }

    Ptensors1<TYPE> view_of(const int i) const{
      return Ptensors1<TYPE>(atoms[i],TENSOR::rows(atoms.offset1(i),atoms.nrows1(i)));
    }

    Ptensors1<TYPE> operator[](const int i) const{
      return Ptensors1<TYPE>(atoms[i],TENSOR::rows(atoms.offset1(i),atoms.nrows1(i)));
    }

    const cnine::Rtensor3_view view3(const int K) const{
      int nc=get_nc();
      return split0(TENSOR::view2(),dim(0)/K,K);
      //return cnine::Rtensor3_view(const_cast<float*>(get_arr()),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
    }

    const cnine::Rtensor3_view view3(const int K, const int offs, const int nc) const{
      int _nc=get_nc();
      return split0(TENSOR::view2().cols(offs,nc),dim(0)/K,K);
      //return cnine::Rtensor3_view(const_cast<float*>(get_arr())+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------
    // these are hacks for the sake of BatchedSubgraphLayer1. todo: introduce constk


    Ptensors<TYPE> reduce0(const int K) const{
      TimedFn T("BatchedSubgraphLayer1b","reduce0",*this);
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      Ptensors<TYPE> R({dim(0)/K,get_nc()},0,get_dev());
      view3(K).sum1_into(R.view2());
      return R;
    }

    Ptensors<TYPE> reduce0(const int K, const int offs, const int nc) const{
      TimedFn T("BatchedSubgraphLayer1b","reduce0",*this);
      //cnine::using_vram_manager vv(ptens_session->managed_gmem);
      Ptensors<TYPE> R({dim(0)/K,nc},0,get_dev());
      view3(K,offs,nc).sum1_into(R.view2());
      return R;
    }

    void broadcast0(const int K, const Ptensors<TYPE>& x, const int offs){
      TimedFn T("BatchedSubgraphLayer1b","broadcast0",*this);
      PTENS_ASSRT(x.ndims()==2);
      view3(K,offs,x.dim(1))+=cnine::repeat1(x.view2(),K);
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors1<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors1<TYPE> R(x.atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps(x.view_of(i));
      //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps(x.view_of(i));});
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps_back(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps_back(x.view_of(i));
      //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps_back(x.view_of(i));});
    }


  public: // ---- Gather -------------------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors1<TYPE> gather(const BatchedAtomsPack<1>& a, const SOURCE& x, const int min_overlaps=1){
      BatchedPtensors1<TYPE> R(a,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_gather(x,BatchedLayerMap::overlaps_map(a,x.atoms));
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensors<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors1<TYPE> gather(const BatchedAtomsPack<1>& a, const SOURCE& x, const BatchedLayerMap& map){
      BatchedPtensors1<TYPE> R(a,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_gather(x,map);
      return R;
    }

    template<typename SOURCE>
      void add_gather(const SOURCE& x, const BatchedLayerMap& map){
      int nc=x.get_nc();

      if constexpr(std::is_same<SOURCE,BatchedPtensors0<TYPE> >::value){
	auto pmap=BatchedGatherPlanFactory::gather_map0(map,atoms,x.atoms,1,x.getk());
	broadcast0(x.reduce0(pmap.in()),pmap.out(),0);
      }

      if constexpr(std::is_same<SOURCE,BatchedPtensors1<TYPE> >::value){
	auto pmap0=BatchedGatherPlanFactory::gather_map0(map,atoms,x.atoms,1,x.getk());
	auto pmap1=BatchedGatherPlanFactory::gather_map1(map,atoms,x.atoms,1,x.getk());
	broadcast0(x.reduce0(pmap0.in()),pmap0.out(),0);
	broadcast1(x.reduce1(pmap1.in()),pmap1.out(),nc);
      }

      if constexpr(std::is_same<SOURCE,BatchedPtensors2<TYPE> >::value){
	auto pmap0=BatchedGatherPlanFactory::gather_map0(map,atoms,x.atoms,1,x.getk());
	auto pmap1=BatchedGatherPlanFactory::gather_map1(map,atoms,x.atoms,1,x.getk());
	broadcast0(x.reduce0(pmap0.in()),pmap0.out(),0);
	broadcast1(x.reduce1(pmap1.in()),pmap1.out(),2*nc);
      }

    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x, const BatchedLayerMap& map){
      int nc=get_nc();

      if constexpr(std::is_same<OUTPUT,BatchedPtensors0<TYPE> >::value){
	auto pmap=BatchedGatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),1);
	broadcast0(x.reduce0(pmap.out()),pmap.in(),0);
      }

      if constexpr(std::is_same<OUTPUT,BatchedPtensors1<TYPE> >::value){
	auto pmap0=BatchedGatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),1);
	auto pmap1=BatchedGatherPlanFactory::gather_map1(map,x.atoms,atoms,x.getk(),1);
	broadcast0(x.reduce0(pmap0.out(),0,nc),pmap0.in(),0);
	broadcast1(x.reduce1(pmap1.out(),nc,nc),pmap1.in(),0);
      }

      if constexpr(std::is_same<OUTPUT,BatchedPtensors2<TYPE> >::value){
	auto pmap0=BatchedGatherPlanFactory::gather_map0(map,x.atoms,atoms,x.getk(),1);
	auto pmap1=BatchedGatherPlanFactory::gather_map1(map,x.atoms,atoms,x.getk(),1);
	broadcast0(x.reduce0_shrink(pmap0.out(),0,nc),pmap0.in(),0);
	broadcast1(x.reduce1_shrink(pmap1.out(),2*nc,nc),pmap1.in(),2*nc);
      }

    }

    
  public: // ---- Indexed Reductions -------------------------------------------------------------------------


    TENSOR reduce0(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors1","reduce0",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce0(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }
    
    TENSOR reduce1(const BatchedAindexPackB& map, const int offs=0, int nc=0) const{
      TimedFn T("BatchedPtensors1","reduce1",*this,map,map.size()*get_nc());
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      TENSOR R({map.nrows,nc},0,get_dev());
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).add_reduce1(R.rows(tail,map[i].nrows),map[i],offs);
	tail+=map[i].nrows;
      }
      return R;
    }
    

  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const TENSOR& x, const BatchedAindexPackB& map, const int offs=0){
      int tail=0;
      for(int i=0; i<size(); i++){
	view_of(i).broadcast0(x.rows(tail,map[i].nrows),map[i],offs);
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


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedPtensors1";
    }

    string repr() const{
      return "<BatchedPtensors1[N="+to_string(size())+",nrows="+to_string(TENSOR::dim(0))+",nc="+to_string(get_nc())+"]>";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Batch "<<i<<":"<<endl;
	oss<<(*this)[i].str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors1& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

//     template<typename SOURCE>
//       void add_gather(const SOURCE& x, const BatchedPtensorMap& map){
//       int nc=x.get_nc();
// 	if constexpr(std::is_same<SOURCE,BatchedPtensors0<TYPE> >::value)
// 	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
// 	if constexpr(std::is_same<SOURCE,BatchedPtensors1<TYPE> >::value){
// 	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
// 	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),nc);
// 	}
// 	if constexpr(std::is_same<SOURCE,BatchedPtensors2<TYPE> >::value){
// 	  broadcast0(x.reduce0(map.atoms(),map.in()),map.out(),0);
// 	  broadcast1(x.reduce1(map.atoms(),map.in()),map.out(),2*nc);
// 	}
//     }

//     template<typename OUTPUT>
//     void add_gather_back(const OUTPUT& x, const BatchedPtensorMap& map){
//       int nc=get_nc();
//       if constexpr(std::is_same<OUTPUT,BatchedPtensors0<TYPE> >::value)
// 	broadcast0(x.reduce0(map.atoms(),map.out()),map.in());
//       if constexpr(std::is_same<OUTPUT,BatchedPtensors1<TYPE> >::value){
// 	broadcast0(x.reduce0(map.atoms(),map.out(),0,nc),map.in());
// 	broadcast1(x.reduce1(map.atoms(),map.out(),nc,nc),map.in());
//       }
//       if constexpr(std::is_same<OUTPUT,BatchedPtensors2<TYPE> >::value){
// 	broadcast0(x.reduce0_shrink(map.atoms(),map.out(),0,nc),map.in());
// 	broadcast1(x.reduce1_shrink(map.atoms(),map.out(),2*nc,nc),map.in());
//       }
//     }

