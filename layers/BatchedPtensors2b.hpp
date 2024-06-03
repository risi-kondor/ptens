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

#ifndef _ptens_BatchedPtensors2b
#define _ptens_BatchedPtensors2b

#include "diff_class.hpp"
#include "object_pack.hpp"

#include "BatchedAtomsPack.hpp"
#include "BatchedAtomsPackN.hpp"
#include "Ptensors2b.hpp"
#include "BatchedPtensorsb.hpp"
#include "SubgraphLayerb.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors2bBatch;
  template<typename TYPE> class Ptensors2bBatch;


  template<typename TYPE>
  class BatchedPtensors2b: public BatchedPtensorsb<TYPE>,
			   public cnine::diff_class<BatchedPtensors2b<TYPE> >{
  public:

    typedef BatchedPtensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef BatchedAtomsPackN<AtomsPack2obj<int> > BatchedAtomsPack2;
    
    using cnine::diff_class<BatchedPtensors2b<TYPE> >::grad;
    using BASE::get_dev;

    BatchedAtomsPackN<AtomsPack2obj<int> > atoms;


    ~BatchedPtensors2b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedPtensors2b(){}

    BatchedPtensors2b(const BatchedAtomsPack& _atoms, const TENSOR& M):
      BASE(M.copy()), atoms(BatchedAtomsPack2(_atoms)){}

    BatchedPtensors2b(const BatchedAtomsPack& _atoms, const cnine::Tensor<float>& M):
      BASE(cnine::Ltensor<float>(M).copy()), atoms(BatchedAtomsPack2(_atoms)){}

    BatchedPtensors2b(const BatchedAtomsPack2& _atoms, const TENSOR& M):
      BASE(M.copy()), atoms(_atoms){}

    BatchedPtensors2b(const BatchedAtomsPack2& _atoms, const int _nc, const int _dev):
      BatchedPtensors2b(_atoms,_nc,0,_dev){}

    BatchedPtensors2b(const BatchedAtomsPack2& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.tsize(),_nc},fcode,_dev), atoms(_atoms){}

    BatchedPtensors2b(const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors2b(BatchedAtomsPack2(_atoms),_nc,0,_dev){}

    BatchedPtensors2b(const BatchedAtomsPack& _atoms, const int _nc, const int fcode, const int _dev):
      BatchedPtensors2b(BatchedAtomsPack2(_atoms),_nc,fcode,_dev){}


    BatchedPtensors2b(const initializer_list<Ptensors2b<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPack2obj<int> > > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPackN<AtomsPack2obj<int> >(x);
    }
	

  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int nc=1;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    BatchedPtensors2b(const BatchedAtomsPack& _atoms, const Args&... args):
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


    BatchedPtensors2b copy() const{
      return BatchedPtensors2b(atoms,TENSOR::copy());
    }

    BatchedPtensors2b copy(const int _dev) const{
      return BatchedPtensors2b(atoms,TENSOR::copy(_dev));
    }

    BatchedPtensors2b zeros_like() const{
      return BatchedPtensors2b(atoms,TENSOR::zeros_like());
    }

    BatchedPtensors2b gaussian_like() const{
      return BatchedPtensors2b(atoms,TENSOR::gaussian_like());
    }

    static BatchedPtensors2b zeros_like(const BatchedPtensors2b& x){
      return BatchedPtensors2b(x.atoms,x.TENSOR::zeros_like());
    }

    static BatchedPtensors2b zeros_like(const BatchedPtensors2b& x, const int nc){
      return BatchedPtensors2b(x.atoms,TENSOR({x.dim(0),nc},0,get_dev()));
    }

    static BatchedPtensors2b gaussian_like(const BatchedPtensors2b& x){
      return BatchedPtensors2b(x.atoms,x.TENSOR::gaussian_like());
    }

    static BatchedPtensors2b* new_zeros_like(const BatchedPtensors2b& x){
      return new BatchedPtensors2b(x.atoms,x.TENSOR::zeros_like());
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    BatchedPtensors2b(const TENSOR& x, const BatchedAtomsPack2& _atoms):
      BASE(x),
      atoms(_atoms){}


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

    BatchedAtomsPack get_atoms() const{
      return atoms.obj->get_atoms();
    }

    BatchedPtensors2b& get_grad(){
      return cnine::diff_class<BatchedPtensors2b<TYPE> >::get_grad();
    }

    const BatchedPtensors2b& get_grad() const{
      return cnine::diff_class<BatchedPtensors2b<TYPE> >::get_grad();
    }

    Ptensors2b<TYPE> view_of(const int i) const{
      return Ptensors2b<TYPE>(TENSOR::rows(atoms.offset(i),atoms.nrows(i)),atoms.obj->obj[i]);
    }

    Ptensors2b<TYPE> operator[](const int i){
      return Ptensors2b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i)),atoms.nrows(i));
    }

    Ptensors2b<TYPE> operator[](const int i) const{
      return Ptensors2b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i),atoms.nrows(i)));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors2b<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors2b<TYPE> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors2b<TYPE> gather(const SOURCE& x, const BatchedAtomsPack& a){
      BatchedPtensors2b<TYPE> R(a,x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
      R.add_gather(x);
      return R;
    }


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps(x.view_of(i));
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    void add_linmaps_back(const SOURCE& x){
      for(int i=0; i<size(); i++)
	view_of(i).add_linmaps_back(x.view_of(i));
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x){
      //(atoms.overlaps_mmap(x.atoms))(*this,x);
      for(int i=0; i<size(); i++)
	view_of(i).add_gather(x.view_of(i));
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      //x.atoms.inverse_overlaps_mmap(atoms)(*this,x);
      for(int i=0; i<size(); i++)
	view_of(i).add_gather_back(x.view_of(i));
    }

    template<typename OUTPUT>
    void add_gather_back_alt(const OUTPUT& x){
      int N=size();
      PTENS_ASSRT(N==x.size());
      x.backward_program(get_grad(),x.get_grad());
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedPtensors2b";
    }

    string repr() const{
      return "<BatchedPtensors2b[N="+to_string(size())+",nrows="+to_string(TENSOR::dim(0))+",nc="+to_string(get_nc())+"]>";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++)
	oss<<(*this)[i]<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors2b& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

