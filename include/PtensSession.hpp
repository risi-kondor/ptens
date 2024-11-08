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

      //cout<<banner()<<endl;
      cout<<"Starting ptens..."<<endl;

    }

    ~PtensSession(){
      //cout<<banner()<<endl;
    }


  public: // ---- Access -----------------------------------------------------------------------------------------


    //    void row_level_operations(const bool x){
    //ptens_global::row_level_operations=x;
    //}

    void cache_atomspack_cats(const bool x){
      ptens_global::cache_atomspack_cats=x;
    }

    //void cache_overlap_maps(const bool x){
    //ptens_global::cache_overlap_maps=x;
    //}

    //void cache_rmaps(const bool x){
    //ptens_global::cache_rmaps=x;
    //}


  public: // ---- I/O --------------------------------------------------------------------------------------------


    string on_off(const bool b) const{
      if(b) return " ON";
      return "OFF";
    }

    string size_or_off(const bool b, const int x) const{
      if(!b) return "\b\bOFF";
      auto s=to_string(x);
      return string('\b',s.length())+s;
    }

    string print_size(const int x) const{
      auto s=to_string(x);
      return string('\b',s.length())+s;
    }

    string banner() const{
      bool with_cuda=0;
      #ifdef _WITH_CUDA
      with_cuda=1;
      #endif

      ostringstream oss;
      oss<<"-------------------------------------"<<endl;
      oss<<"Ptens 0.0 "<<endl;
      //cout<<endl;
      oss<<"CUDA support:                     "<<on_off(with_cuda)<<endl;
      //oss<<"Row level gather operations:      "<<on_off(ptens_global::row_level_operations)<<endl;
      oss<<"-------------------------------------"<<endl;
      return oss.str();
    }
    
    string status_str() const{
      bool with_cuda=0;
      #ifdef _WITH_CUDA
      with_cuda=1;
      #endif

      ostringstream oss;
      oss<<"---------------------------------------"<<endl;
      oss<<" Ptens 0.0 "<<endl;
      //cout<<endl;
      oss<<" CUDA support:                     "<<on_off(with_cuda)<<endl;
      //oss<<" Row level gather operations:      "<<on_off(ptens_global::row_level_operations)<<endl;
      oss<<endl;
      oss<<" AtomsPack cat cache:               "<<
	size_or_off(ptens_global::cache_atomspack_cats, ptens_global::atomspack_cat_cache.size())<<endl;
      oss<<" Overlap maps cache:                "<<
      print_size(ptens_global::overlaps_maps_cache.size())<<endl;
      oss<<" Gather plans cache:                "<<
      print_size(ptens_global::gather_plan_cache.size())<<endl;
      oss<<endl;
      oss<<" Batched AtomsPack cat cache        "<<
	print_size(ptens_global::batched_atomspack_cat_cache.size())<<endl;
      oss<<" Batched overlap maps cache:        "<<
      print_size(ptens_global::batched_overlaps_maps_cache.size())<<endl;
      oss<<" Batched_gather plans cache:        "<<
      print_size(ptens_global::batched_gather_plan_cache.size())<<endl;
      oss<<endl;
      oss<<" Graph cache:                        "<<
	print_size(ptens_global::graph_cache.size())<<endl;
      oss<<" Graph elist cache:                  "<<
	print_size(ptens_global::graph_cache.edge_list_map.size())<<endl;
      oss<<" Subgraph cache:                     "<<
	print_size(ptens_global::subgraph_cache.size())<<endl;
      oss<<"---------------------------------------"<<endl;
      return oss.str();
    }
    
     string str() const{
      return banner();
    }

    friend ostream& operator<<(ostream& stream, const PtensSession& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
