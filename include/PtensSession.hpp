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

#include <fstream>
#include <chrono>
#include <ctime>
#include <unordered_set>

#include "CnineSession.hpp"
#include "object_bank.hpp"

#include "Ptens_base.hpp"
#include "SubgraphObj.hpp"
#include "GgraphCache.hpp"


namespace ptens{


  class PtensSession{
  public:

    cnine::cnine_session* cnine_session=nullptr;

    ofstream logfile;
    //cnine::object_bank<Subgraph,SubgraphObj> subgraph_bank([]
    //(const Subgraph& x){return new SubgraphObj(x);});
    std::unordered_set<SubgraphObj> subgraphs;
    GgraphCache graph_cache;


    PtensSession(const int _nthreads=1){

      cnine_session=new cnine::cnine_session(_nthreads);

      #ifdef _WITH_CUDA
      cout<<"Initializing ptens with GPU support."<<endl;
      #else
      cout<<"Initializing ptens without GPU support."<<endl;
      #endif


      logfile.open("ptens.log");
      auto time = std::chrono::system_clock::now();
      std::time_t timet = std::chrono::system_clock::to_time_t(time);
      #ifdef _WITH_CUDA
      logfile<<"Ptens session started with CUDA at "<<std::ctime(&timet)<<endl;
      #else
      logfile<<"Ptens session started without CUDA at "<<std::ctime(&timet)<<endl;
      #endif 
      
    }


    ~PtensSession(){

      cout<<"Shutting down ptens."<<endl;
      std::time_t timet = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      logfile<<endl<<"Ptens session shut down at "<<std::ctime(&timet)<<endl<<endl<<endl;
      logfile.close();
      
      delete cnine_session;
    }


  public: // Logging 

    void log(const string msg){
      std::time_t timet = std::time(nullptr);
      char os[30];
      strftime(os,30,"%H:%M:%S ",std::localtime(&timet));
      logfile<<os<<msg<<endl;
    }
    
    template<typename OBJ>
    void log(const string msg, const OBJ& obj){
      std::time_t timet = std::time(nullptr);
      char os[30];
      strftime(os,30,"%H:%M:%S ",std::localtime(&timet));
      logfile<<os<<msg<<obj.repr()<<endl;
    }
    
  };

}


#endif 
