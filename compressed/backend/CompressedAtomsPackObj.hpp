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


#ifndef _ptens_CompressedAtomsPackObj
#define _ptens_CompressedAtomsPackObj

#include "AtomsPackObj.hpp"
#include "Ltensor.hpp"
#include "ColumnSpace.hpp"
#include "observable.hpp"


namespace ptens{

  class CompressedAtomsPackObj: public cnine::observable<CompressedAtomsPackObj>{
  public:

    //typedef AtomsPackObj BASE;
    typedef cnine::Ltensor<float> TENSOR;


    shared_ptr<AtomsPackObj> atoms;
    cnine::Ltensor<float> bases; 
    // for each Ptensors, bases is a k x nvecs matrix


  public: // ---- Constructors -------------------------------------------------------------------------------


    CompressedAtomsPackObj():
      observable(this){}
    
    CompressedAtomsPackObj(const shared_ptr<AtomsPackObj>& _atoms, const TENSOR& M):
      observable(this),
      atoms(_atoms),
      bases(M.copy()){
      PTENS_ASSRT(bases.ndims()==2 || bases.ndims()==3);
      if(bases.ndims()==2){
	bases.dims=bases.dims.insert(0,atoms->size());
	bases.strides=bases.strides.insert(0,0);
      }
    }

    //CompressedAtomsPackObj(const TENSOR& M, const shared_ptr<AtomsPackObj>& _atoms):
    //observable(this),
    //atoms(_atoms),
    //bases(M.unsqueeze(0).broadcast(0,_atoms->size()).fuse({0,1})){}

    CompressedAtomsPackObj(const shared_ptr<AtomsPackObj>& _atoms, const int _nvecs, const int fcode, const int _dev=0):
      observable(this),
      atoms(_atoms),
      bases({nrows1(),_nvecs},fcode,_dev){
      if(fcode==4){
	for(int i=0; i<size(); i++){
	  cnine::Ltensor<float> M=cnine::ColumnSpace(basis(i));
	  basis(i).cols(0,M.dim(1))=M;
	}
      }
    }

	  
  public: // ---- Named constructors -------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    int constk() const{
      return atoms->constk;
    }

    int size() const{
      return atoms->size();
    }

    int nvecs() const{
      return bases.dims.back();
    }

    int row_offset1(const int i) const{
      return atoms->row_offset1(i);
    }

    int nrows1() const{
      return atoms->nrows1();
    }

    int nrows1(const int i) const{
      return atoms->nrows1(i);
    }

    Atoms atoms_of(const int i) const{
      return (*atoms)[i];
    }

    TENSOR basis(const int i) const{
      if(bases.ndims()==3)
	return bases.slice(1,i);
      return bases.rows(row_offset1(i),nrows1(i));
    }

    bool operator==(const CompressedAtomsPackObj& x) const{
      if((atoms.get()!=x.atoms.get())&&(*atoms!=*(x.atoms))) return false;
      return bases==x.bases;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "CompressedAtomsPackObj";
    }

    string repr() const{
      return "<SepctralAtomsPackObj n="+to_string(size())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<size(); i++){
	oss<<basis(i);
	if(i<size()-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CompressedAtomsPackObj& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
