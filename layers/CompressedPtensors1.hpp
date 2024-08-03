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

#ifndef _ptens_Ptensors1
#define _ptens_Ptensors1

#include "diff_class.hpp"
#include "Ptensors1.hpp"
#include "CompressedPtensors.hpp"
#include "BlockCsparseMatrix.hpp"


namespace ptens{

  #ifdef _WITH_CUDA
  #endif 


  template<typename TYPE>
  class CompressedPtensors1: public CompressedPtensors<TYPE>, public cnine::diff_class<Ptensors1<TYPE> >{
  public:

    //friend class Ptensors0<TYPE>;
    //friend class CompressedPtensors2<TYPE>;

    typedef Ptensors<TYPE> BASE;
    typedef cnine::Ltensor<TYPE> TENSOR;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;

    using cnine::diff_class<Ptensors1<TYPE> >::grad;

    using BASE::get_dev;
    using TENSOR::dim;
    using TENSOR::move_to_device;
    using TENSOR::add;
    using TENSOR::dev;
    using TENSOR::strides;
    using TENSOR::get_arr;
    using TENSOR::cols;

    using BASE::nc;
    using BASE::atoms;
    using BASE::size;
    using BASE::atoms_of;
    using BASE::get_nc;


    ~Ptensors1(){
#ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    int constk() const{
      return _atoms.constk();
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------


    template<typename SOURCE>
    static CompressedPtensors1<float> linmaps(const SpectralAtomsPack& _atoms, const SOURCE& x){
      CompressedPtensors1<float> R(_atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    void add_linmaps(const Ptensors0<TYPE>& x){
      broadcast0(x);
    }

    void add_linmaps(const Ptensors1<TYPE>& x){
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
      return "CompressedPtensors1";
    }

    string repr() const{
      return "<CompressedPtensors1[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      if(get_dev()>0){
	CompressedPtensors1 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<(*this)(i).str(indent);
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CompressedPtensors1& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
