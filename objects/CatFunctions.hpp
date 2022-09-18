#ifndef _CatFunctions
#define _CatFunctions

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "SparseMx.hpp"

namespace ptens{

  void add_concat(Ptensors1& r, const Ptensors0& x, const SparseMatrix& G){
    int nc=source.get_nc();
    assert(dest.nc==nc);
    int n=G.n;
    int m=G.m;
    assert(r.size()==n);
    assert(x.size()==m);

    G.forall_nonzero([&](const int i, const int j, const float v){
	Atoms ratoms=r.atoms_of(i);
	Atoms xatoms=x.atoms_of(j);
	Atoms intersect=ratoms.intersect(xatoms);
	intersect.foreach([&](const int p){
	    int a=ratoms(p);
	    int b=xatoms(p);
	    for(int c=0; c<nc; c++)
	      dest.view2_of(i).add(ratoms(j),c,source.view1_of(j)(c));
	  });
      });
  }


  concat1(const AtomsPack& _atoms, const SparseMatrix& G, const Ptensors0& x){
    Ptensors1 R=Ptensors1::zero(_atoms,x.get_nc());
    add_concat(R,x,G);
    return R;
  }
  

}

#endif 
