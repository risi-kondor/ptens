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
#ifndef _ConcatenatingLayer0to1
#define _ConcatenatingLayer0to1

#include "Ptensor0pack.hpp"
#include "Ptensor1pack.hpp"


namespace ptens{

  class ConcatenatingLayer0to1{
  public:

    ConcatenatingLayer0to1(){}

  public:

    void forward(Ptensor1pack& dest, const Ptensor0pack& source){
      int nc=source.get_nc();
      assert(dest.nc==nc);
      int N=dest.size();

      for(int i=0; i<N; i++){
	Atoms dest_atoms=dest.atoms_of(i);
	for(auto j:dest_atoms){
	  Atoms source_atoms=source.atoms_of(j);
	  Atoms intersect=dest_atoms.intersect(source_atoms);
	  intersect.foreach([&](const int p){
	      int a=source_atoms(p);
	      int b=dest_atoms(p);
	      for(int c=0; c<nc; c++)
		dest.view2_of(i).set(dest_atoms(j),c,source.view1_of(j)(c));
	    });
	  }
	}
	
    }

  };

}

#endif
