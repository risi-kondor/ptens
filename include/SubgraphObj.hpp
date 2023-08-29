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

#ifndef _ptens_SubgraphObj
#define _ptens_SubgraphObj

#include "Ptens_base.hpp"
#include "SparseRmatrix.hpp"
#include "Hgraph.hpp"


namespace ptens{


  class SubgraphObj: public Hgraph /*public cnine::SparseRmatrix*/{
  public:

    typedef Hgraph BASE;
    typedef cnine::SparseRmatrix BaseMatrix;


    //using BaseMatrix::BaseMatrix;
    using BASE::BASE;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SubgraphObj(const int _n):
      SubgraphObj(_n,_n){}

    SubgraphObj(const int _n, const initializer_list<pair<int,int> >& list): 
      SubgraphObj(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    //SubgraphObj(const cnine::RtensorA& A):
    //SubgraphObj(A){}

    SubgraphObj(const int n, const cnine::RtensorA& M):
      SubgraphObj(n){
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.get_dim(0)==2);
      PTENS_ASSRT(M.max()<n);
      int nedges=M.get_dim(1);
      for(int i=0; i<nedges; i++)
	set(M(0,i),M(1,i),1.0);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "SubgraphObj";
    }

  };

}


namespace std{
  template<>
  struct hash<ptens::SubgraphObj>{
  public:
    size_t operator()(const ptens::SubgraphObj& x) const{
      return hash<cnine::SparseRmatrix>()(x);
    }
  };
}


#endif 
