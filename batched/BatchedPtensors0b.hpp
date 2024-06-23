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

#ifndef _ptens_BatchedPtensors0b
#define _ptens_BatchedPtensors0b

#include "diff_class.hpp"
#include "object_pack.hpp"

#include "BatchedAtomsPack.hpp"
#include "BatchedAtomsPackN.hpp"
#include "Ptensors0b.hpp"
#include "BatchedPtensorsb.hpp"
#include "MultiLoop.hpp"

namespace ptens{

  template<typename TYPE> class Ptensors1bBatch;
  template<typename TYPE> class Ptensors2bBatch;


  template<typename TYPE>
  class BatchedPtensors0b: public BatchedPtensorsb<TYPE>,
			   public cnine::diff_class<BatchedPtensors0b<TYPE> >{
  public:

    typedef BatchedPtensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef BatchedAtomsPackN<AtomsPack0obj<int> > BatchedAtomsPack0;
    
    using cnine::diff_class<BatchedPtensors0b<TYPE> >::grad;
    using BASE::get_dev;

    BatchedAtomsPackN<AtomsPack0obj<int> > atoms;

    cnine::GatherMapProgramPack forward_program;
    cnine::GatherMapProgramPack backward_program;


    ~BatchedPtensors0b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedPtensors0b(){}

    BatchedPtensors0b(const TENSOR& M, const vector<int> sizes):
      BASE(M.copy()){
	  vector<shared_ptr<AtomsPack0obj<int> > > x;
      for(auto p:sizes) x.push_back(to_share(new AtomsPack0obj<int>(p)));
      atoms=BatchedAtomsPackN<AtomsPack0obj<int> >(x);
    }

    BatchedPtensors0b(const BatchedAtomsPack& _atoms, const cnine::Tensor<float>& M):
      BASE(M.copy()), atoms(BatchedAtomsPack0(_atoms)){}

    BatchedPtensors0b(const BatchedAtomsPack0& _atoms, const TENSOR& M):
      BASE(M.copy()), atoms(_atoms){}

    BatchedPtensors0b(const BatchedAtomsPack0& _atoms, const int _nc, const int _dev):
      BatchedPtensors0b(_atoms,_nc,0,_dev){}

    BatchedPtensors0b(const BatchedAtomsPack0& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.tsize(),_nc},fcode,_dev), atoms(_atoms){}

    BatchedPtensors0b(const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors0b(BatchedAtomsPack0(_atoms),_nc,0,_dev){}

    BatchedPtensors0b(const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev):
      BatchedPtensors0b(BatchedAtomsPack0(_atoms),_nc,fcode,_dev){}


    BatchedPtensors0b(const initializer_list<Ptensors0b<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPack0obj<int> > > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPackN<AtomsPack0obj<int> >(x);
    }
	
    /*
    BatchedPtensors0b(const vector<const TENSOR&> M):
      BASE(cnine::Ltensor<TYPE>::stack(0,M)){
      vector<shared_ptr<AtomsPack0obj<int> > > x;
      for(auto& p:M) x.push_back(to_share(new AtomsPack0obj<int>(p.dim(0))));
      atoms=BatchedAtomsPackN<AtomsPack0obj<int> >(x);
    }
    */

  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    BatchedPtensors0b(const BatchedAtomsPack& _atoms, const Args&... args):
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


    BatchedPtensors0b copy() const{
      return BatchedPtensors0b(atoms,TENSOR::copy());
    }

    BatchedPtensors0b copy(const int _dev) const{
      return BatchedPtensors0b(atoms,TENSOR::copy(_dev));
    }

    BatchedPtensors0b zeros_like() const{
      return BatchedPtensors0b(atoms,TENSOR::zeros_like());
    }

    BatchedPtensors0b gaussian_like() const{
      return BatchedPtensors0b(atoms,TENSOR::gaussian_like());
    }

    static BatchedPtensors0b zeros_like(const BatchedPtensors0b& x){
      return BatchedPtensors0b(x.TENSOR::zeros_like(),x.atoms);
    }

    static BatchedPtensors0b zeros_like(const BatchedPtensors0b& x, const int nc){
      return BatchedPtensors0b(TENSOR({x.dim(0),nc},0,get_dev()),x.atoms);
    }

    static BatchedPtensors0b gaussian_like(const BatchedPtensors0b& x){
      return BatchedPtensors0b(x.atoms,x.TENSOR::gaussian_like());
    }

    static BatchedPtensors0b* new_zeros_like(const BatchedPtensors0b& x){
      return new BatchedPtensors0b(x.atoms,x.TENSOR::zeros_like());
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    BatchedPtensors0b(const TENSOR& x, const BatchedAtomsPack0& _atoms):
      BASE(x),
      atoms(_atoms){}


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

    BatchedAtomsPack get_atoms() const{
      return atoms.obj->get_atoms();
    }

    BatchedPtensors0b& get_grad(){
      return cnine::diff_class<BatchedPtensors0b<TYPE> >::get_grad();
    }

    const BatchedPtensors0b& get_grad() const{
      return cnine::diff_class<BatchedPtensors0b<TYPE> >::get_grad();
    }

    Ptensors0b<TYPE> view_of(const int i) const{
      return Ptensors0b<TYPE>(TENSOR::rows(atoms.offset(i),atoms.nrows(i)),atoms.obj->obj[i]);
    }

    Ptensors0b<TYPE> operator[](const int i){
      return Ptensors0b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i)),atoms.nrows(i));
    }

    Ptensors0b<TYPE> operator[](const int i) const{
      return Ptensors0b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i),atoms.nrows(i)));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors0b<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors0b<TYPE> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors0b<TYPE> gather(const SOURCE& x, const BatchedAtomsPack& a, const int min_overlaps=1){
      BatchedPtensors0b<TYPE> R(a,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_gather(x,min_overlaps);
      return R;
    }


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps(const SOURCE& x){
      //for(int i=0; i<size(); i++)
      //view_of(i).add_linmaps(x.view_of(i));
      cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps(x.view_of(i));});
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps_back(const SOURCE& x){
      //for(int i=0; i<size(); i++)
      //view_of(i).add_linmaps_back(x.view_of(i));
      cnine::MultiLoop(size(),[&](const int i){view_of(i).add_linmaps_back(x.view_of(i));});
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x,const int min_overlaps=1){
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


    static string classname(){
      return "BatchedPtensors0b";
    }

    string repr() const{
      return "<BatchedPtensors0b[N="+to_string(size())+",nrows="+to_string(TENSOR::dim(0))+",nc="+to_string(get_nc())+"]>";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++)
	oss<<(*this)[i]<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors0b& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

