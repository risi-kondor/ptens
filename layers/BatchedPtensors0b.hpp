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


    ~BatchedPtensors0b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedPtensors0b(){}

    BatchedPtensors0b(const BatchedAtomsPack0& _atoms, const TENSOR& M):
      BASE(M), atoms(_atoms){}

    BatchedPtensors0b(const BatchedAtomsPack0& _atoms, const int _nc, const int _dev):
      BatchedPtensors0b(_atoms,_nc,0,_dev){}

    BatchedPtensors0b(const BatchedAtomsPack0& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.tsize(),_nc},fcode,_dev), atoms(_atoms){}


    BatchedPtensors0b(const initializer_list<Ptensors0b<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPack0obj<int> > > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPackN<AtomsPack0obj<int> >(x);
    }
	

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
      return BatchedPtensors0b(x.atoms,x.TENSOR::zeros_like());
    }

    static BatchedPtensors0b zeros_like(const BatchedPtensors0b& x, const int nc){
      return BatchedPtensors0b(x.atoms,x.TENSOR({TENSOR::dim(0),nc},0,get_dev()));
    }

    static BatchedPtensors0b gaussian_like(const BatchedPtensors0b& x){
      return BatchedPtensors0b(x.atoms,x.TENSOR::gaussian_like());
    }

    static BatchedPtensors0b* new_zeros_like(const BatchedPtensors0b& x){
      return new BatchedPtensors0b(x.atoms,x.TENSOR::zeros_like());
    }
    

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
      BatchedPtensors0b<TYPE> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors0b<TYPE> gather(const SOURCE& x, const BatchedAtomsPack& a){
      BatchedPtensors0b<TYPE> R(a,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
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

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedPtensors0b";
    }

    string repr() const{
      return "BatchedPtensors0b";
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

