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

#ifndef _ptens_CompundTransferMapObj
#define _ptens_CompundTransferMapOBj

#include "AtomsPackObj.hpp"


namespace ptens{

  inline vector<int> decode(const AindexPack& pack, const AtomsPackObj& atoms){
    int Nmessages=out_ixpack.size();
    int N=tmap.in->tail;
    vector<int> r(N);

    int c=0;
    for(int i=0; i<Nmessages; i++){
      int t=pack.tix(i);
      int n=pack.nix(i);
      for(int j=0; j<n; j++)
	r[c++]=atoms.decode1(t,j);
    }

    return r;
  }



  class TransferMapInstruction{
  public:

  };


  class Reduce1to0: public TransferMapInstruction{
  public:

    GatherMapB operator()(const AtomsPackObj& in_atoms, const AtomsPackObj& out_atoms, const TransferMap& tmap){
      return GatherMapB(decode(*tmap.in,in_atoms),decode(*tmap.out,out_atoms));
    }

  }


 class Reduce1to1: public TransferMapInstruction{
  public:

    GatherMapB operator()(const AtomsPackObj& in_atoms, const AtomsPackObj& out_atoms, const TransferMap& tmap){
      return GatherMapB(decode(*tmap.in,in_atoms),decode(*tmap.out,out_atoms));
    }

  }






  class TransferMapProgram{

  };

}

#endif 


	/*
      AindexPack& in_ixpack=*tmap.in;
      AindexPack& out_ixpack=*tmap.out;
      int Nmessages=out_ixpack.size();

      unordered_map<int,int> counts;
      for(int i=0; i<Nmessages; i++){
	int offs=out_atoms.offset(out_ixpack.tix(i));
	for(int j=0; j<out_ixpack.nix(i); j++)
	  counts[offs+out_ixpack(i,j)]++;
      }
      int Noutputs=counts.size();

      map<int,vector<int> > count_bins;
      for(auto p:counts)
	count_bins[-p.second].push_back(p.first);


      GatherMapB G(sizes);
      for(int m=0; m<Nmessages; m++){
	int offs=out_atoms.offset(out_ixpack.tix(i));
	for(int j=0; j<out_ixpack.nix(i); j++)
	  G.push_back(mapping
offs+out_ixpack(i,j)]++;
	
      }

      int
      tmap.for_each_edge([&](const int i, const int j, const float v){
	  for
	});
	*/
 
