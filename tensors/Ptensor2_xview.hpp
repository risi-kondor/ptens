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
#ifndef _Ptensor2_xview
#define _Ptensor2_xview

#include "Rtensor2_view.hpp"
#include "Ptensor1_xview.hpp"


namespace ptens{

  class Ptensor2_xview: public cnine::Rtensor3_view{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;


  public:

    const vector<int> ix;


  public:

    Ptensor2_xview(float* _arr, const int _nc, const int _s0, const int _s1, const int _s2, 
      const vector<int>& _ix, const int _dev=0): 
      Rtensor3_view(_arr,_ix.size(),_ix.size(),_nc,_s0,_s1,_s2,_dev),
      ix(_ix){}


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_regular() const{
      return false;
    }

    //float operator()(const int i0, const int i1) const{
    //return arr[s0*ix[i0]+s1*i1];
    //}

    // TODO: rewrite to avoid virtual fns
    float operator()(const int i0, const int i1, const int i2) const{
      return arr[s0*ix[i0]+s1*ix[i1]+s2*i2];
    }

    float get(const int i0, const int i1, const int i2) const{
      return arr[s0*ix[i0]+s1*ix[i1]+s2*i2];
    }

    void set(const int i0, const int i1, const int i2, float x) const{
      arr[s0*ix[i0]+s1*ix[i1]+s2*i2]=x;
    }

    void inc(const int i0, const int i1, const int i2, float x) const{
      arr[s0*ix[i0]+s1*ix[i1]+s2*i2]+=x;
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    void sum0_into(const Rtensor2_view& r){
      CNINE_CPUONLY();
      assert(r.n0==n1);
      assert(r.n1==n2);
      for(int i1=0; i1<n1; i1++)
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i0=0; i0<n0; i0++) 
	    t+=arr[s0*ix[i0]+s1*ix[i1]+s2*i2];
	  r.inc(i1,i2,t);
	}
    }

    void sum1_into(const Rtensor2_view& r){
      CNINE_CPUONLY();
      assert(r.n0==n0);
      assert(r.n1==n2);
      for(int i0=0; i0<n0; i0++) 
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i1=0; i1<n1; i1++)
	    t+=arr[s0*ix[i0]+s1*ix[i1]+s2*i2];
	  r.inc(i0,i2,t);
	}
    }

    void sum01_into(const Rtensor1_view& r){
      CNINE_CPUONLY();
      assert(r.n0==n2);
      for(int i2=0; i2<n2; i2++){
	float t=0; 
	for(int i0=0; i0<n0; i0++) 
	  for(int i1=0; i1<n1; i1++)
	    t+=arr[s0*ix[i0]+s1*ix[i1]+s2*i2];
	r.inc(i2,t);
      }
    }


  public: // ---- Other views -------------------------------------------------------------------------------


    Ptensor1_xview diag01() const{
      return Ptensor1_xview(arr,n2,s0+s1,s2,ix,dev);
    }

    Ptensor2_xview transp() const{
      return Ptensor2_xview(arr,n2,s1,s0,s2,ix,dev);
    }

  };

}

#endif
