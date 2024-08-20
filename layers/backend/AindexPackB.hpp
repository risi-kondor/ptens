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

#ifndef _ptens_AindexPackB
#define _ptens_AindexPackB

#include <map>

#include "hlists.hpp"
#include "monitored.hpp"
#include "Atoms.hpp"
#include "GatherMapB.hpp"
#include "Ltensor.hpp"
#include "RemoteCopy.hpp"


namespace ptens{


  class AindexPackB: public cnine::TensorView<int>{
  public:

    typedef cnine::TensorView<int> ITENSOR;

    int _max_nix=0;

    int nrows=0;
    int n_input_rows=0;
    int n_gather_lists=0;

    int count1=0;
    int count2=0;

    cnine::GatherMapB gather_map;

    cnine::RemoteCopy<int,ITENSOR> on_device=cnine::RemoteCopy<int,ITENSOR>([this](const int& _dev){
	return to_share(new ITENSOR(*this,_dev));});

    cnine::RemoteCopy<int,ITENSOR> gmap_on_device=cnine::RemoteCopy<int,ITENSOR>([this](const int& _dev){
	return to_share(new ITENSOR(gather_map.arr.to_tensor(_dev)));});


  public: // ---- Constructors ------------------------------------------------------------------------------


    AindexPackB(const int n, const int maxn):
      ITENSOR({n,maxn+4}){}


  public: // ---- Copying -----------------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------

    
    int size() const{
      return ITENSOR::dim(0);
    }

    int toffset(const int i) const{
      return (*this)(i,0);
    }

    int nix(const int i) const{
      return (*this)(i,1);
    }

    int soffset(const int i) const{
      return (*this)(i,2);
    }

    int ssize(const int i) const{
      return (*this)(i,3);
    }

    int ix(const int i, const int j) const{
      return (*this)(i,j+4);
    }

    vector<int> ix(const int i) const{
      int n=nix(i);
      vector<int> R(n);
      for(int j=0; j<n; j++)
	R[j]=(*this)(i,j+4);
      return R;
    }

    void set(const int i, const int _toffset, const int _nix, const int _soffset, const int _ssize, const vector<int> v){
      PTENS_ASSRT(i<dim(0));
      PTENS_ASSRT(v.size()<=dim(1)-4);
      ITENSOR::set(i,0,_toffset);
      ITENSOR::set(i,1,_nix);
      ITENSOR::set(i,2,_soffset);
      ITENSOR::set(i,3,_ssize);
      for(int j=0; j<v.size(); j++)
	ITENSOR::set(i,j+4,v[j]);
    }

    void preload(const int _dev) const{
      on_device(_dev);
      gmap_on_device(_dev);
    }

//     cnine::Rtensor1_view chunk0(const cnine::Ltensor<float>& x, const int i) const{
//       return x.row(toffset(i)).view1();
//     }

//     cnine::Rtensor2_view chunk1(const cnine::Ltensor<float>& x, const int i) const{
//       return x.rows(toffset(i),nix(i)).view2();
//     }
    
//     cnine::Rtensor3_view chunk2(const cnine::Ltensor<float>& x, const int i) const{
//       int k=nix(i);
//       return cnine::split0(x.rows(toffset(i),k*k).view2(),k,k);
//     }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "AindexPackB";
    }
    
    string repr() const{
      return "<AindexPack[N="+std::to_string(size())+"]>";
    }

    friend ostream& operator<<(ostream& stream, const AindexPackB& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
 
 
