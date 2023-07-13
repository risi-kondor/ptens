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

    cnine::cnine_session* cnine_session=nullptr;


    PtensSession(const int _nthreads=1){

      #ifdef _WITH_CUDA
      cout<<"Initializing ptens with GPU support."<<endl;
      #else
      cout<<"Initializing ptens without GPU support."<<endl;
      #endif

      cnine_session=new cnine::cnine_session(_nthreads);

    }


    ~PtensSession(){
      cout<<"Shutting down ptens."<<endl;
      delete cnine_session;
    }
    
  };

}


#endif 
