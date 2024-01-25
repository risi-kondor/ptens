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


namespace ptens{

  template<typename TYPE> class Ptensors1bBatch;
  template<typename TYPE> class Ptensors2bBatch;


  template<typename TYPE>
  class BatchedPtensors1b: public BatchedPtensorsb<TYPE>,
			   public cnine::diff_class<BatchedPtensors1b<TYPE> >{
  public:

    typedef BatchedPtensorsb<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef BatchedAtomsPackN<AtomsPack1obj<int> > BatchedAtomsPack1;
    
    using cnine::diff_class<BatchedPtensors1b<TYPE> >::grad;
    using BASE::get_dev;

    BatchedAtomsPackN<AtomsPack1obj<int> > atoms;


    ~BatchedPtensors1b(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //BatchedPtensors1b(){}

    BatchedPtensors1b(const BatchedAtomsPack1& _atoms, const TENSOR& M):
      BASE(M), atoms(_atoms){}

    BatchedPtensors1b(const BatchedAtomsPack1& _atoms, const int _nc, const int _dev):
      BatchedPtensors1b(_atoms,_nc,0,_dev){}

    BatchedPtensors1b(const BatchedAtomsPack1& _atoms, const int _nc, const int fcode, const int _dev):
      BASE({_atoms.tsize(),_nc},fcode,_dev), atoms(_atoms){}

    BatchedPtensors1b(const BatchedAtomsPack& _atoms, const int _nc, const int _dev):
      BatchedPtensors1b(BatchedAtomsPack1(_atoms),_nc,0,_dev){}


    BatchedPtensors1b(const initializer_list<Ptensors1b<TYPE> >& list):
      BASE(cnine::Ltensor<TYPE>::stack(0,list)){
      vector<shared_ptr<AtomsPack1obj<int> > > x;
      for(auto& p:list) x.push_back(p.atoms.obj);
      atoms=BatchedAtomsPackN<AtomsPack1obj<int> >(x);
    }
	

  public: // ----- Spawning ----------------------------------------------------------------------------------


    BatchedPtensors1b copy() const{
      return BatchedPtensors1b(atoms,TENSOR::copy());
    }

    BatchedPtensors1b copy(const int _dev) const{
      return BatchedPtensors1b(atoms,TENSOR::copy(_dev));
    }

    BatchedPtensors1b zeros_like() const{
      return BatchedPtensors1b(atoms,TENSOR::zeros_like());
    }

    BatchedPtensors1b gaussian_like() const{
      return BatchedPtensors1b(atoms,TENSOR::gaussian_like());
    }

    static BatchedPtensors1b zeros_like(const BatchedPtensors1b& x){
      return BatchedPtensors1b(x.atoms,x.TENSOR::zeros_like());
    }

    static BatchedPtensors1b zeros_like(const BatchedPtensors1b& x, const int nc){
      return BatchedPtensors1b(x.atoms,x.TENSOR({TENSOR::dim(0),nc},0,get_dev()));
    }

    static BatchedPtensors1b gaussian_like(const BatchedPtensors1b& x){
      return BatchedPtensors1b(x.atoms,x.TENSOR::gaussian_like());
    }

    static BatchedPtensors1b* new_zeros_like(const BatchedPtensors1b& x){
      return new BatchedPtensors1b(x.atoms,x.TENSOR::zeros_like());
    }
    

  public: // ----- Conversions -------------------------------------------------------------------------------


    BatchedPtensors1b(const TENSOR& x, const BatchedAtomsPack1& _atoms):
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

    BatchedPtensors1b& get_grad(){
      return cnine::diff_class<BatchedPtensors1b<TYPE> >::get_grad();
    }

    const BatchedPtensors1b& get_grad() const{
      return cnine::diff_class<BatchedPtensors1b<TYPE> >::get_grad();
    }

    Ptensors1b<TYPE> view_of(const int i) const{
      return Ptensors1b<TYPE>(TENSOR::rows(atoms.offset(i),atoms.nrows(i)),atoms.obj->obj[i]);
    }

    //Ptensors1b<TYPE> operator[](const int i){
    //return Ptensors1b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i)),atoms.nrows(i));
    //}

    Ptensors1b<TYPE> operator[](const int i) const{
      return Ptensors1b<TYPE>(atoms.obj->obj[i],TENSOR::rows(atoms.offset(i),atoms.nrows(i)));
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static BatchedPtensors1b<TYPE> linmaps(const SOURCE& x){
      BatchedPtensors1b<TYPE> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<BatchedPtensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors1b<TYPE> gather(const SOURCE& x, const BatchedAtomsPack& a){
      BatchedPtensors1b<TYPE> R(a,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
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


    string classname() const{
      return "BatchedPtensors1b";
    }

    string str(const string indent="") const{ 
      ostringstream oss;
      for(int i=0; i<size(); i++)
	oss<<(*this)[i]<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensors1b& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

