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

#ifndef _ptens_PtensorMessageMapObj
#define _ptens_PtensorMessageMapObj

#include "hlists.hpp"
#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"


namespace ptens{


  class PtensorMessageMapObj: public cnine::hlists<int>{
  public:
    
    typedef cnine::hlists<int> BASE;


  public: // ---- Static constructors -----------------------------------------------------------------------------


    static shared_ptr<PtensorMessageMapObj> all_overlapping(const AtomsPackObj& out, const AtomsPackObj& in){
      cnine::map_of_lists<int,int> A;
      
      if(in.size()<10){
	for(int i=0; i<out.size(); i++){
	  auto v=out(i);
	  for(int j=0; j<in.size(); j++){
	    auto w=in(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      A.push_back(i,j);
	  }
	}
      }
      else{
	cnine::map_of_lists<int,int> memberships;
	in.for_each([&](const int i, const int& p){memberships.push_back(p,i);});
	out.for_each([&](const int i, const int& p){
	    memberships.for_each_in_list(p,[&](const int& j){A.push_back(i,j);});});
      }

      return shared_ptr<PtensorMessageMapObj>(new PtensorMessageMapObj(BASE(A)));
    }


  public: // ---- Conversions --------------------------------------------------------------------------------------
    

    PtensorMessageMapObj(const BASE& x):
      BASE(x){cout<<"c"<<endl;}

    PtensorMessageMapObj(BASE&& x):
      BASE(std::move(x)){}


    string str(const string indent=""){
      return BASE::str();
    }

  };

}

#endif 

