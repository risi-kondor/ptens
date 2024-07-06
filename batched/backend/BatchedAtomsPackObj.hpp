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


#ifndef _ptens_BatchedAtomsPackObj
#define _ptens_BatchedAtomsPackObj

#include "shared_object_pack.hpp"
#include "AtomsPackObj.hpp"


namespace ptens{


  class BatchedAtomsPackObj: public cnine::shared_object_pack<AtomsPackObj>{
  public:

    typedef cnine::shared_object_pack<AtomsPackObj> BASE;

    using BASE::BASE;
    using BASE::size;
    using BASE::operator[];


  public: // ---- Constructors -------------------------------------------------------------------------------


    BatchedAtomsPackObj(const vector<vector<vector<int> > >& v){
      for(auto& p:v)
	push_back(to_share(new AtomsPackObj(p)));
    }

    BatchedAtomsPackObj(const initializer_list<initializer_list<initializer_list<int> > >& v){
      for(auto& p:v)
	push_back(to_share(new AtomsPackObj(p)));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    vector<vector<vector<int > > > as_vecs() const{
      vector<vector<vector<int > > > R;
      for(int i=0; i<size(); i++)
	R.push_back((*this)[i].as_vecs());
      return R;
    }

  };

}

#endif 
