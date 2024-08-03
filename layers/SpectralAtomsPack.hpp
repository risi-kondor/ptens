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

#ifndef _ptens_SpectralAtomsPack
#define _ptens_SpectralAtomsPack

#include "AtomsPack.hpp"


namespace ptens{

  class SpectralAtomsPack: public AtomsPack{
  public:


    typedef cnine::Ltensor<float> TENSOR;


    SpectralAtomsPack(const shared_ptr<SpectralAtomsPackObj>& x):
      obj(x){}

    SpectralAtomsPack(const AtomsPack& _atoms, const int _nvecs, const int _dev=0):
      obj(make_shared<SpectralAtomsPackObj>(_atoms,_nvecs,_dev)){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    SpectralAtomsPack gaussian(const AtomsPack& _atoms, const int _nvecs, const int fcode=0, const int _dev=0){
      int k=_atoms.constk();
      if(k>0)
	return make_shared<SpectralAtomsPackObj>(_atoms.obj,TENSOR::gaussian(cnine::gdims(_atoms.size(),k,fcode,_nvecs),_dev));
      else
	return make_shared<SpectralAtomsPackObj>(_atoms.obj,TENSOR::gaussian(cnine::gdims(_atoms.nrows1(),fcode,_nvecs),_dev));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    SpectralAtomsPackObj& _obj() const{
      return dynamic_cast<SpectralAtomsPackObj&>(*obj);
    }

    TENSOR operator()(const int i) const{
      return _obj(i);
    }

    TENSOR as_matrix() const{
      if(_obj.evecs.ndims()==2) return _obj.evecs;
      return _obj.evecs.fuse({0,1});
    }

    TENSOR as_tensor() const{
      if(_obj.evecs.ndims()==3) return _obj.evecs;
      return _obj.evecs.split(0,constk());
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SpectralAtomsPack";
    }

    string repr() const{
      return "<SpectralAtomsPack n="+to_string(size())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<size(); i++){
	oss<<(*this)(i);
	if(i<size()-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SpectralAtomsPack& v){
      stream<<v.str(); return stream;}


  };

#endif 

