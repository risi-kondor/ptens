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

#ifndef _ptens_CompressedPtensors2
#define _ptens_CompressedPtensors2

#include "diff_class.hpp"
#include "Ptensors2.hpp"
#include "CompressedPtensors.hpp"
#include "BlockCsparseMatrix.hpp"


namespace ptens{

  #ifdef _WITH_CUDA
  #endif 


  template<typename TYPE>
  class CompressedPtensors2: public CompressedPtensors<TYPE>, public cnine::diff_class<CompressedPtensors2<TYPE> >{
  public:

    //friend class Ptensors0<TYPE>;
    //friend class CompressedPtensors2<TYPE>;

    typedef CompressedPtensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;

    using cnine::diff_class<CompressedPtensors2<TYPE> >::grad;

    using BASE::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;
    using TENSOR::dev;
    using TENSOR::strides;
    using TENSOR::get_arr;
    using TENSOR::cols;
    using TENSOR::slice;

    using BASE::nc;
    using BASE::atoms;
    using BASE::size;
    //using BASE::atoms_of;
    using BASE::get_nc;


    ~CompressedPtensors2(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    CompressedPtensors2(const CompressedAtomsPack& _atoms, const int nc, const int fcode=0, const int _dev=0):
      BASE(_atoms,TENSOR({_atoms.size(),_atoms.nvecs(),_atoms.nvecs(),nc},fcode,_dev)){}


  public: // ---- Transport ----------------------------------------------------------------------------------

    
    CompressedPtensors2(const CompressedPtensors2& x, const int _dev):
      BASE(x.atoms,TENSOR(x,_dev)){}


  public: // ---- Conversions --------------------------------------------------------------------------------


    CompressedPtensors2(const CompressedAtomsPack& _atoms, const Ptensors2<TYPE>& x):
      BASE(_atoms,TENSOR({_atoms.size(),_atoms.nvecs(),_atoms.nvecs(),x.get_nc()},0,x.get_dev())){
      PTENS_ASSRT(*x.atoms.obj==*atoms->atoms);
      int N=size();
      for(int i=0; i<N; i++)
	(*this)(i).add_mprod((*atoms)(i).transp(), x(i)*((*atoms)(i))); // TODO 
    }

    Ptensors2<TYPE> uncompress(){
      Ptensors2<TYPE> R(AtomsPack(atoms->atoms),get_nc(),get_dev());
      int N=size();
      for(int i=0; i<N; i++)
	R(i).add_mprod((*atoms)(i)*(*this)(i),(*atoms)(i)); // TODO 
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //int constk() const{
    //return _atoms.constk();
    //}

    TENSOR operator()(const int i) const{
      return slice(0,i);
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static CompressedPtensors2 linmaps(const CompressedAtomsPack& _atoms, const SOURCE& x){
      CompressedPtensors2 R(_atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const Ptensors2<TYPE>& x){
      int nc=x.get_nc();
      broadcast0(x.reduce0());
      cols(nc,nc)+=x;
    }

  public: // ---- Reductions ---------------------------------------------------------------------------------

  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const TENSOR& X, const int offs=0){
      //as_matrix
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "CompressedPtensors2";
    }

    string repr() const{
      return "<CompressedPtensors2[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	CompressedPtensors2 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<(*this)(i).str(indent);
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CompressedPtensors2& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
