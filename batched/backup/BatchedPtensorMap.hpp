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

#ifndef _ptens_BatchedPtensorMap
#define _ptens_BatchedPtensorMap

#include "object_pack_s.hpp"
#include "PtensorMap.hpp"
#include "BatchedAtomsPack.hpp"
#include "BatchedAindexPack.hpp"

namespace ptens{


  class BatchedPtensorMap{ //: public cnine::object_pack_s<PtensorMapObj>{
  public:

    BatchedAindexPack in_indices;
    BatchedAindexPack out_indices;
    BatchedAtomsPack<0> _atoms;


  public: // ---- Named constructors ------------------------------------------------------------------------


  public: // ---- Access ------------------------------------------------------------------------------------


    const BatchedAtomsPack<0>& atoms() const{
      return _atoms;
    }

    const BatchedAindexPack& in() const{
      return in_indices;
    }

    const BatchedAindexPack& out() const{
      return out_indices;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    template<int k, int p>
    static BatchedPtensorMap overlaps(const BatchedAtomsPack<k>& out, const BatchedAtomsPack<p>& in){
      BatchedPtensorMap R;
      int N=out.size();

      vector<PtensorMap> maps;
      for(int i=0; i<N; i++)
	maps.push_back(PtensorMapFactory::overlaps(out[i],in[i]));
      //maps.push_back(PtensorMap::overlaps_map(out[i],in[i]));
	  
      vector<AtomsPack> atomsv;
      for(int i=0; i<N; i++)
	atomsv.push_back(maps[i].atoms());
      R._atoms=BatchedAtomsPack<0>(atomsv);

      vector<shared_ptr<AindexPack> > inv;
      for(int i=0; i<N; i++)
	inv.push_back(maps[i].obj->in);
      R.in_indices=BatchedAindexPack(inv);

      vector<shared_ptr<AindexPack> > outv;
      for(int i=0; i<N; i++)
	outv.push_back(maps[i].obj->out);
      R.out_indices=BatchedAindexPack(outv);

      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedPtensrMap";
    }

    string repr() const{
      return "BatchedPtensorMap";
    }

    string str(const string indent="") const{
      return "";
    }

    friend ostream& operator<<(ostream& stream, const BatchedPtensorMap& v){
      stream<<v.str(); return stream;}



  };

}

#endif 
