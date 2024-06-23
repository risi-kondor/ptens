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

#ifndef _ptens_BatchedPtensors1b
#define _ptens_BatchedPtensors1b

#include "diff_class.hpp"
#include "object_pack.hpp"

#include "BatchedAtomsPack.hpp"
#include "BatchedAtomsPackN.hpp"
#include "Ptensors1b.hpp"
#include "BatchedPtensorsb.hpp"
#include "MultiLoop.hpp"


namespace ptens{

  //template<typename TYPE> class Ptensors1bBatch;
  //template<typename TYPE> class Ptensors2bBatch;


  template<typename TYPE>
  class BatchedPtensors1b: public BatchedPtensorsb<TYPE>,
			   public cnine::diff_class<BatchedPtensors1b<TYPE> >{
  public:

    typedef BatchedPtensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef BatchedAtomsPackN<AtomsPack1obj<int> > BatchedAtomsPack1;
    
    using cnine::diff_class<BatchedPtensors1b<TYPE> >::grad;
    using BASE::get_dev;

    using TENSOR::dim;
    using TENSOR::get_arr;

    BatchedAtomsPackN<AtomsPack1obj<int> > atoms;

    cnine::GatherMapProgramPack forward_program;
    cnine::GatherMapProgramPack backward_program;


    ~BatchedPtensors1b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedPtensors1b(){}

    BatchedPtensors1b(const BatchedAtomsPack& _atoms, const cnine::Ltensor<float>& M):
      BASE(M.copy(ptens_session->managed_gmem)), atoms(BatchedAtomsPack1(_atoms)){}

    BatchedPtensors1b(const BatchedAtomsPack& _atoms, const cnine::Tensor<float>& M):
      BASE(cnine::Ltensor<float>(M).copy(ptens_session->managed_gmem)), atoms(BatchedAtomsPack1(_atoms)){}

    BatchedPtensors1b(const BatchedAtomsPack1& _atoms, const TENSOR& M):
      BASE(M.copy(ptens_session->managed_gmem)), atoms(_atoms){}

    BatchedPtensors1b(const BatchedAtomsPack1& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.tsize(),_nc},fcode,_dev), atoms(_atoms){}


    BatchedPtensors1b(const BatchedAtomsPack1& _atoms, const int _nc, const int _dev):
      BatchedPtensors1b(_atoms,_nc,0,_dev){}

    BatchedPtensors1b(const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors1b(BatchedAtomsPack1(_atoms),_nc,0,_dev){}

    BatchedPtensors1b(const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev):
      BatchedPtensors1b(BatchedAtomsPack1(_atoms),_nc,fcode,_dev){}


    BatchedPtensors1b(TYPE* _arr, const BatchedAtomsPack1& _atoms, const int _nc, const int _dev):
      BASE(_arr,{_atoms.tsize(),_nc},_dev), atoms(_atoms){}

    BatchedPtensors1b(TYPE* _arr, const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors1b(_arr,BatchedAtomsPack1(_atoms),_nc,_dev){}


    // TODO 
    BatchedPtensors1b(const initializer_list<Ptensors1b<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPack1obj<int> > > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPackN<AtomsPack1obj<int> >(x);
    }
	
    static BatchedPtensors1b cat(const vector<BatchedPtensors1b>& list){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      PTENS_ASSRT(list.size()>0);
      vector<BatchedAtomsPack1> v;
      for(auto& p:list)
	v.push_back(p.atoms);
      BatchedPtensors1b R(BatchedAtomsPack1::cat(v),list[0].get_nc(),list[0].get_dev());

      int N=list[0].size();
      for(int i=0; i<N; i++){
	vector<cnine::Ltensor<TYPE> > w;
	for(int j=0; j<list.size(); j++)
	  w.push_back(list[j].view_of(i));
	R.view_of(i).cnine:: template Ltensor<TYPE>::operator=(cnine::Ltensor<TYPE>::stack(0,w));
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
    BatchedPtensors1b(const BatchedAtomsPack& _atoms, const Args&... args):
      atoms(_atoms){
      vparams v;
      unroller(v,args...);
      BASE::reset(BASE({atoms.tsize(),v.nc},v.fcode,v.dev));
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


    BatchedPtensors1b copy() const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(TENSOR::copy(),atoms);
    }

    BatchedPtensors1b copy(const int _dev) const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(TENSOR::copy(_dev),atoms);
    }

    BatchedPtensors1b zeros_like() const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(TENSOR::zeros_like(),atoms);
    }

    BatchedPtensors1b zeros_like(const int nc) const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(TENSOR({dim(0),nc},0,get_dev()),atoms);
    }

    BatchedPtensors1b gaussian_like() const{
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(TENSOR::gaussian_like(),atoms);
    }

    static BatchedPtensors1b zeros_like(const BatchedPtensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(x.BASE::zeros_like(),x.atoms);
    }

    static BatchedPtensors1b zeros_like(const BatchedPtensors1b& x, const int nc){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(TENSOR({x.dim(0),nc},0,get_dev()),x.atoms);
    }

    static BatchedPtensors1b gaussian_like(const BatchedPtensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return BatchedPtensors1b(x.TENSOR::gaussian_like(),x.atoms);
    }

    static BatchedPtensors1b* new_zeros_like(const BatchedPtensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new BatchedPtensors1b(x.BASE::zeros_like(),x.atoms);
    }
    
    static BatchedPtensors1b* new_like(TYPE* _arr, const BatchedPtensors1b& x){
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      return new BatchedPtensors1b(x.TENSOR::like(_arr),x.atoms);
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    BatchedPtensors1b(const TENSOR& x, const BatchedAtomsPack1& _atoms):
      BASE(x),
      atoms(_atoms){}


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

    BatchedAtomsPack get_atoms() const{
      return atoms.obj->get_atoms();
    }

    BatchedPtensors1b& get_grad(){ // why do we need these?
      return cnine::diff_class<BatchedPtensors1b<TYPE> >::get_grad();
    }

    const BatchedPtensors1b& get_grad() const{
      return cnine::diff_class<BatchedPtensors1b<TYPE> >::get_grad();
    }

    BatchedPtensors1b& get_grad(TYPE* _arr){
      if(!grad) grad=new_like(_arr,*this);
      return *grad;
    }

    Ptensors1b<TYPE> view_of(const int i) const{
      return Ptensors1b<TYPE>(TENSOR::rows(atoms.offset(i),atoms.nrows(i)),atoms.obj->obj[i]);
    }

    //Ptensors1b<TYPE> operator[](const int i){
    //return Ptensors1b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i)),atoms.nrows(i));
    //}

    Ptensors1b<TYPE> operator[](const int i) const{
      return Ptensors1b<TYPE>(AtomsPack1(atoms.obj->obj[i]),TENSOR::rows(atoms.offset(i),atoms.nrows(i)));
    }

    const cnine::Rtensor3_view view3(const int K) const{
      int nc=get_nc();
      return cnine::Rtensor3_view(const_cast<float*>(get_arr()),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
    }

    cnine::Rtensor3_view view3(const int K){
      int nc=get_nc();
      return cnine::Rtensor3_view(get_arr(),dim(0)/K,K,nc,K*nc,nc,1,get_dev());
    }

    const cnine::Rtensor3_view view3(const int K, const int offs, const int nc) const{
      int _nc=get_nc();
      return cnine::Rtensor3_view(const_cast<float*>(get_arr())+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
    }

    cnine::Rtensor3_view view3(const int K, const int offs, const int nc){
      int _nc=get_nc();
      return cnine::Rtensor3_view(get_arr()+offs,dim(0)/K,K,nc,K*_nc,_nc,1,get_dev());
    }


    static int nrows(const BatchedAtomsPack& _atoms){
      return BatchedAtomsPack1(_atoms).tsize();
    }

    //template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    //static int nchannels(const SOURCE& x){
    //return x.get_nc()*vector<int>({1,2,5})[x.getk()];
    //}


  public: // ---- Linmaps ------------------------------------------------------------------------------------

    // these are hacks for the sake of BatchedSubgraphLayer1. todo: introduce constk


    Ptensorsb<TYPE> reduce0(const int K) const{
      TimedFn T("BatchedSubgraphLayer1b","reduce0",*this);
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      Ptensorsb<TYPE> R({dim(0)/K,get_nc()},0,get_dev());
      view3(K).sum1_into(R.view2());
      return R;
    }

    Ptensorsb<TYPE> reduce0(const int K, const int offs, const int nc) const{
      TimedFn T("BatchedSubgraphLayer1b","reduce0",*this);
      cnine::using_vram_manager vv(ptens_session->managed_gmem);
      Ptensorsb<TYPE> R({dim(0)/K,nc},0,get_dev());
      view3(K,offs,nc).sum1_into(R.view2());
      return R;
    }

    void broadcast0(const int K, const Ptensorsb<TYPE>& x, const int offs){
      TimedFn T("BatchedSubgraphLayer1b","broadcast0",*this);
      PTENS_ASSRT(x.ndims()==2);
      view3(K,offs,x.dim(1))+=cnine::repeat1(x.view2(),K);
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors1b<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors1b<TYPE> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps(x.view_of(i));
      //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps(x.view_of(i));});
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps_back(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps_back(x.view_of(i));
      //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps_back(x.view_of(i));});
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors1b<TYPE> gather(const SOURCE& x, const BatchedAtomsPack& a, const int min_overlaps=1){
      BatchedPtensors1b<TYPE> R(a,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_gather(x,min_overlaps);
      return R;
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x, const int min_overlaps=1){
      int N=size();
      PTENS_ASSRT(N==x.size());
      for(int i=0; i<N; i++){
	MessageList mlist=atoms.obj->obj[i]->atoms->overlaps_mlist(*x.atoms.obj->obj[i]->atoms,min_overlaps);
	MessageMap mmap=atoms.obj->obj[i]->message_map(*mlist.obj,*x.atoms.obj->obj[i]);
	forward_program.obj.push_back(mmap.obj);
	backward_program.obj.push_back(to_share(new cnine::GatherMapProgram(mmap.obj->inv()))); // eliminate the copy here 
      }
      forward_program(*this,x);
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      //x.atoms.inverse_overlaps_mmap(atoms)(*this,x);
      //for(int i=0; i<size(); i++)
      //view_of(i).add_gather_back(x.view_of(i));
      //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_gather_back(x.view_of(i));});
      int N=size();
      PTENS_ASSRT(N==x.size());
      cnine::GatherMapProgramPack P;
      for(int i=0; i<N; i++){
	MessageList mlist=x.atoms.obj->obj[i]->atoms->overlaps_mlist(*atoms.obj->obj[i]->atoms);
	MessageMap mmap=x.atoms.obj->obj[i]->message_map(*mlist.obj,*atoms.obj->obj[i]);
	P.obj.push_back(to_share(new cnine::GatherMapProgram(mmap.obj->inv()))); // eliminate the copy here 
      }
      P(*this,x);
    }

    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){
      x.backward_program(get_grad(),x.get_grad());
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedPtensors1b";
    }

    string repr() const{
      return "<BatchedPtensors1b[N="+to_string(size())+",nrows="+to_string(TENSOR::dim(0))+",nc="+to_string(get_nc())+"]>";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Batch "<<i<<":"<<endl;
	oss<<(*this)[i].str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors1b& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

    /*
    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      //x.atoms.inverse_overlaps_mmap(atoms)(*this,x);
      //for(int i=0; i<size(); i++)
      //view_of(i).add_gather_back(x.view_of(i));
      //cnine::MultiLoop(size(),[&](const int i){view_of(i).add_gather_back(x.view_of(i));});
      int N=size();
      PTENS_ASSRT(N==x.size());
      //cnine::GatherMapProgramPack P;
      for(int i=0; i<N; i++){
	MessageList mlist=x.atoms.obj->obj[i]->atoms->overlaps_mlist(*atoms.obj->obj[i]->atoms);
	MessageMap mmap=x.atoms.obj->obj[i]->message_map(*mlist.obj,*atoms.obj->obj[i]);
	backward_program.obj.push_back(to_share(new cnine::GatherMapProgram(mmap.obj->inv()))); // eliminate the copy here 
      }
      backward_program(*this,x);
    }
    */

    /*
    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){
      int N=size();
      PTENS_ASSRT(N==x.size());
      cnine::GatherMapProgramPack P;
      for(int i=0; i<N; i++){
	P.obj.push_back(to_share(new cnine::GatherMapProgram(mmap.obj->inv()))); // eliminate the copy here 
      }
      P(*this,x);
    }
    */
