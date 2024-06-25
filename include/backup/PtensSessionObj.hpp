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

#ifndef _PtensSessionObj
#define _PtensSessionObj

#include <fstream>
#include <chrono>
#include <ctime>
#include <unordered_set>

#include "CnineSession.hpp"
#include "object_bank.hpp"
#include "MemoryManager.hpp"

#include "Ptens_base.hpp"
#include "SubgraphObj.hpp"
#include "GgraphCache.hpp"
#include "OverlapsMmapCache.hpp"

//namespace ptens{
//class PtensSessionObj;
//}

//extern ptens::PtensSessionObj* ptens::ptens_session;


namespace ptens{

  //extern OverlapsMessageMapBank* overlaps_bank;


  class PtensSessionObj{
  public:

    cnine::cnine_session* cnine_session=nullptr;

    bool row_gathers=false;

    ofstream logfile;
    std::unordered_set<SubgraphObj> subgraphs;
    GgraphCache graph_cache;
    cnine::MemoryManager* managed_gmem=nullptr;

    //bool cache_overlap_maps=false;


    PtensSessionObj(const int _nthreads=1){

      cnine_session=new cnine::cnine_session(_nthreads);

      //ptens::overlaps_bank=new OverlapsMessageMapBank();

      cout<<banner()<<endl;

      logfile.open("ptens.log");
      auto time = std::chrono::system_clock::now();
      std::time_t timet = std::chrono::system_clock::to_time_t(time);
#ifdef _WITH_CUDA
      logfile<<"Ptens session started with CUDA at "<<std::ctime(&timet)<<endl;
#else
      logfile<<"Ptens session started without CUDA at "<<std::ctime(&timet)<<endl;
#endif 
      
    }


    ~PtensSessionObj(){

      cout<<"Shutting down ptens."<<endl;
      if(managed_gmem) delete managed_gmem;
      std::time_t timet = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      logfile<<endl<<"Ptens session shut down at "<<std::ctime(&timet)<<endl<<endl<<endl;
      logfile.close();
      
      delete cnine_session;
    }

  public: // ---- Access ----------------------------------------------------------------------------------------


    //void set_cache_overlap_maps(const bool x){
    //cache_overlap_maps=x;
    //}


  public: // ---- Logging ---------------------------------------------------------------------------------------


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


  public: // ---- I/O ---------------------------------------------------------------------------------------


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
      oss<<"Row level gather operations:      "<<on_off(row_gathers)<<endl;
      oss<<endl;
      oss<<"Overlap maps cache:               "<<size_or_off(cache_overlap_maps,overlaps_cache.rmemsize())<<endl;
      oss<<"-------------------------------------"<<endl;
      return oss.str();
    }
    
  };


}


#endif 
