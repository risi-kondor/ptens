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

#ifndef _Ptens_GraphPreloader
#define _Ptens_GraphPreloader

#include "Ggraph.hpp"
#include "AtomsPackObj.hpp"
#include "GatherPlanObj.hpp"


namespace ptens{


  class GgraphPreloader{
  public:

    shared_ptr<GgraphObj> G;
    
    GgraphPreloader(const Ggraph& _G):
      G(_G.obj){}


  public: // -------------------------------------------------------------------------------------------------


    void preload(const int dev){
      for(auto& p:G->subgraphpack_cache){
	for(auto& p2:p.second.obj->related_gatherplans){
	  if(p2.expired()) continue;
	  auto plan=p2.lock();
	  plan->in_map->preload(dev);
	  plan->out_map->preload(dev);
	}
      }
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------
    
    
    string classname() const{
      return "GgraphPreloader";
    }
    
    string repr() const{
      return "<GgraphPreloader>";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"Preloader for "<<G->repr()<<endl;
      for(auto& p:G->subgraphpack_cache){
	auto& atoms=*p.second.obj;
	oss<<"  "<<atoms.repr()<<endl;
	//for(auto& p:atoms.related_layermaps){
	//auto lmap=p.lock();
	//oss<<"    "<<lmap->repr()<<endl;
	//}
	for(auto& p:atoms.related_gatherplans){
	  if(p.expired()) continue;
	  auto plan=p.lock();
	  oss<<"    "<<plan->repr()<<endl;
	  oss<<"      "<<plan->in_map->repr()<<endl;
	  oss<<"      "<<plan->out_map->repr()<<endl;
	}
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GgraphPreloader& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
