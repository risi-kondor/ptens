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
 */


#ifndef _ptens_SpectralAtomsPackObj
#define _ptens_SpectralAtomsPackObj

#include "AtomsPackObj.hpp"
#include "Ltensor.hpp"


namespace ptens{

  class SpectralAtomsPackObj: public AtomsPackObj{
  public:

    typedef AtomsPackObj BASE;
    typedef cnine::Ltensor<float> TENSOR;


    cnine::Ltensor<float> evecs;

  public:
    
    SpectralAtomsPackObj(const AtomsPackObj& _atoms, const TENSOR& M):
      BASE(_atoms),
      evecs(M){
      if(constk>0){
	PTENS_ASSRT(evecs.ndims()==3);
	PTENS_ASSRT(evecs.dim(0)==size());
	PTENS_ASSRT(evecs.dim(1)==constk);
      }else{
	PTENS_ASSRT(evecs.ndims()==2);
	PTENS_ASSRT(evecs.dim(1)==nrows1());
      }
    }

    SpectralAtomsPackObj(const AtomsPackObj& _atoms, const int _nvecs, const int fcode, const int _dev=0):
      BASE(_atoms){
      if(constk>0)
	evecs.reset(TENSOR(cnine::dim(size(),constk,_nvecs),fcode,_dev));
      else
	evecs.reset(TENSOR(cnine::dim(nrows1(),_nvecs),fcode,_dev));
    }

	  
  public: // ---- Named constructors -------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    int nvecs() const{
      return evecs.dims.back();
    }

    Ltensor<float> operator()(const int i){
      if(evecs.ndims()==3)
	return evecs.slice(1,i);
      return evecs.rows(row_offset1(i),nrows1(i));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SpectralAtomsPackObj";
    }

    string repr() const{
      return "<SepctralAtomsPackObj n="+to_string(size())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const AtomsPackObj& v){
      stream<<v.str(); return stream;}}


  };

}

#endif 
