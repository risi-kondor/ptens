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

#ifndef _PtensSession
#define _PtensSession

#include "CnineSession.hpp"


namespace ptens{


  class PtensSession{
  public:

    cnine::cnine_session  cnineSession;


    PtensSession(const int _nthreads=1):
      cnineSession(_nthreads){

      cout<<banner()<<endl;

    }

    ~PtensSession(){
    }

  public: // ---- Access -----------------------------------------------------------------------------------------


    void cache_overlap_maps(const bool x){
      ptens_global::cache_overlap_maps=x;
    }


  public: // ---- I/O --------------------------------------------------------------------------------------------


    string on_off(const bool b) const{
      if(b) return " ON";
      return "OFF";
    }

    string size_or_off(const bool b, const int x) const{
      if(!b) return "OFF";
      return to_string(x);
    }

    string banner() const{
      bool with_cuda=0;
      #ifdef _WITH_CUDA
      with_cuda=1;
      #endif

      ostringstream oss;
      oss<<"-------------------------------------"<<endl;
      oss<<"Ptens 0.0 "<<endl;
      cout<<endl;
      oss<<"CUDA support:                     "<<on_off(with_cuda)<<endl;
      oss<<"Row level gather operations:      "<<on_off(ptens_global::row_level_operations)<<endl;
      oss<<endl;
      oss<<"Overlap maps cache:               "<<
	size_or_off(ptens_global::cache_overlap_maps, ptens_global::overlaps_cache.rmemsize())<<endl;
      oss<<"-------------------------------------"<<endl;
      return oss.str();
    }
    
    string str() const{
      return banner();
    }

    friend ostream& operator<<(ostream& stream, const PtensSession& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
