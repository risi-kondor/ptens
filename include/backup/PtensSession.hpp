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

#include "PtensSessionObj.hpp"

//extern ptens::PtensSessionObj* ptens::ptens_session;


namespace ptens{



  class PtensSession{
  public:

    PtensSession(const int _nthreads=1){
      assert(!ptens_session);
      ptens_session=new PtensSessionObj(_nthreads);
    }

    ~PtensSession(){
      delete ptens_session;
    }

  public: // ---- Access -----------------------------------------------------------------------------------------


    void cache_overlap_maps(const bool x){
      ptens::cache_overlap_maps=x;
    }


  public: // ---- I/O --------------------------------------------------------------------------------------------


    string banner() const{
      return ptens_session->banner();
    }

    string str() const{
      return ptens_session->banner();
    }

    friend ostream& operator<<(ostream& stream, const PtensSession& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
