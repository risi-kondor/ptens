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

#ifndef _ConcatFunctions
#define _ConcatFunctions

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "SparseRmatrix.hpp"
#include "Hgraph.hpp"

namespace ptens{

  void add_concat(Ptensors1& r, const Ptensors0& x, const Hgraph& G, bool self=0){
    int nc=x.get_nc();
    assert(r.nc==nc);
    int n=G.n;
    int m=G.m;
    assert(r.size()==n);
    assert(x.size()==m);
    if(self) assert(n==m);
    
    if(self){
      for(int i=0; i<n; i++)
	r.view2_of(i).slice0(r.atoms_of(i)(i))+=x.view1_of(i);
    }

    G.forall_nonzero([&](const int i, const int j, const float v){
	Atoms ratoms=r.atoms_of(i);
	Atoms xatoms=x.atoms_of(j);
	Atoms intersect=ratoms.intersect(xatoms);
	intersect.foreach([&](const int p){
	    int a=ratoms(p);
	    int b=xatoms(p);
	    r.view2_of(i).slice0(a)+=x.view1_of(j);
	    //for(int c=0; c<nc; c++)
	    //r.view2_of(i).inc(a,c,x.view1_of(j)(c));
	  });
      });
  }


  void add_concat_back(Ptensors0& r, const Ptensors1& x, const Hgraph& G, bool self=0){
    int nc=x.get_nc();
    assert(r.nc==nc);
    int n=G.n;
    int m=G.m;
    assert(r.size()==n);
    assert(x.size()==m);
    if(self) assert(n==m);
    
    if(self){
      for(int i=0; i<n; i++)
	x.view1_of(i)+=r.view2_of(i).slice0(r.atoms_of(i)(i));
    }

    G.forall_nonzero([&](const int i, const int j, const float v){
	Atoms ratoms=r.atoms_of(i);
	Atoms xatoms=x.atoms_of(j);
	Atoms intersect=ratoms.intersect(xatoms);
	intersect.foreach([&](const int p){
	    int a=ratoms(p);
	    int b=xatoms(p);
	    x.view1_of(j)+=r.view2_of(i).slice0(a);
	  });
      });
  }


  Ptensors1 concat1(const AtomsPack& _atoms, const Hgraph& G, const Ptensors0& x){
    Ptensors1 R=Ptensors1::zero(_atoms,x.get_nc());
    add_concat(R,x,G);
    return R;
  }
  
  Ptensors1 concat(const Ptensors0& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.nhoods(1),x.get_nc());
    add_concat(R,x,G,1);
    return R;
  }
  

}

#endif 
