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

#ifndef _ptens_PtensorMap
#define _ptens_PtensorMap

#include "PtensorMapObj.hpp"
#include "AtomsPack.hpp"


namespace ptens{

  //class AtomsPackObj;


  class PtensorMap{
  public:
    
    shared_ptr<PtensorMapObj> obj;

    PtensorMap(){
      PTENS_ASSRT(false);}

    PtensorMap(const shared_ptr<PtensorMapObj>& x):
      obj(x){}

    PtensorMap(PtensorMapObj* x):
      obj(x){}

    /*
    static PtensorMap overlaps_map(const AtomsPack& out, const AtomsPack& in){
      if(ptens_global::cache_overlap_maps) 
	return PtensorMap(ptens_global::overlaps_cache(out,in));
      return new PtensorMapObj(*in.obj,*out.obj); 
    }
    */


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      return obj->is_empty();
    }

    bool is_graded() const{
      return obj->is_graded();
    }

    const AtomsPack atoms() const{
      return obj->atoms;
    }

    const AindexPack& in() const{
      return *obj->in;
    }

    const AindexPack& out() const{
      return *obj->out;
    }

    std::shared_ptr<cnine::GatherMapB> get_bmap() const{
      return obj->get_bmap();
    }
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "PtensorMap";
    }

    string repr() const{
      return "PtensorMap";
    }

    string str(const string indent="") const{
      return obj->str();
    }

    friend ostream& operator<<(ostream& stream, const PtensorMap& v){
      stream<<v.str(); return stream;}


  };

}

#endif 

