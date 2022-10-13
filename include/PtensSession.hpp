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
