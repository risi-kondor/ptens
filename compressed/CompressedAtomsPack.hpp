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

#ifndef _ptens_CompressedAtomsPack
#define _ptens_CompressedAtomsPack

#include "AtomsPack.hpp"
#include "CompressedAtomsPackObj.hpp"


namespace ptens{

  class CompressedAtomsPack{
  public:

    typedef cnine::Ltensor<float> TENSOR;

    shared_ptr<CompressedAtomsPackObj> obj;


  public: // ---- Constructors -------------------------------------------------------------------------------


    CompressedAtomsPack(const shared_ptr<CompressedAtomsPackObj>& x):
      obj(x){}

    CompressedAtomsPack(const AtomsPack& _atoms, const TENSOR& M):
      obj(cnine::to_share(new CompressedAtomsPackObj(_atoms.obj,M))){}

    CompressedAtomsPack(const AtomsPack& _atoms, const int _nvecs, const int fcode, const int _dev=0):
      obj(cnine::to_share(new CompressedAtomsPackObj(_atoms.obj,_nvecs,fcode,_dev))){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    //CompressedAtomsPack
    
    CompressedAtomsPack gaussian(const AtomsPack& _atoms, const int _nvecs, const int fcode=0, const int _dev=0){
      int k=_atoms.constk();
      if(k>0)
	return cnine::to_share(new CompressedAtomsPackObj(_atoms.obj,TENSOR(cnine::dims(_atoms.size(),k,fcode,_nvecs),4,_dev)));
      else
	return cnine::to_share(new CompressedAtomsPackObj(_atoms.obj,TENSOR(cnine::dims(_atoms.nrows1(),fcode,_nvecs),4,_dev)));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //CompressedAtomsPackObj& _obj() const{
    //return dynamic_cast<CompressedAtomsPackObj&>(*obj);
    //}

    int constk() const{
      return obj->constk();
    }

    int size() const{
      return obj->size();
    }

    int nvecs() const{
      return obj->nvecs();
    }

    AtomsPack atoms() const{
      return obj->atoms;
    }

    Atoms atoms(const int i) const{
      return (*obj->atoms)[i];
    }

    TENSOR basis(const int i) const{
      return obj->basis(i);
    }

    TENSOR as_matrix() const{
      if(obj->bases.ndims()==2) return obj->bases;
      return obj->bases.fuse({0,1});
    }

    TENSOR as_tensor() const{
      if(obj->bases.ndims()==3) return obj->bases;
      return obj->bases.split(0,constk());
    }

    bool operator==(const CompressedAtomsPack& x) const{
      if(obj.get()==x.obj.get()) return true;
      return (*obj)==(*x.obj);
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "CompressedAtomsPack";
    }

    string repr() const{
      return "<CompressedAtomsPack n="+to_string(size())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<atoms(i)<<":"<<endl;
	oss<<basis(i).to_string(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CompressedAtomsPack& v){
      stream<<v.str(); return stream;}


  };

}

#endif 

