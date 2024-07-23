/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 */

#ifndef _Ptens_BatchedGgraph
#define _Ptens_BatchedGgraph

#include "BatchedGgraphObj.hpp"
#include "Ggraph.hpp"


namespace ptens{


  class BatchedGgraph{
  public:

    typedef BatchedGgraphObj OBJ;


    shared_ptr<OBJ> obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    //BatchedGgraph():
    //obj(new BatchedGgraphObj()){}

    BatchedGgraph(BatchedGgraphObj* _obj):
      obj(_obj){}

    BatchedGgraph(shared_ptr<BatchedGgraphObj> _obj):
      obj(_obj){}

    BatchedGgraph(const vector<Ggraph>& x):
      BatchedGgraph(new BatchedGgraphObj(cnine::mapcar<Ggraph,shared_ptr<GgraphObj> >
	  (x,[](const Ggraph& y){return y.obj;}))){}

    BatchedGgraph(const vector<int>& keys):
      BatchedGgraph(new BatchedGgraphObj(keys)){}


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static BatchedGgraph from_edge_list(const vector<int>& sizes, const cnine::TensorView<int>& M){
      return BatchedGgraphObj::from_edge_list_p(sizes,M);
    }

    static BatchedGgraph from_edge_list(const cnine::TensorView<int>& M, vector<int>& indicators){ //, const bool cached=false){
      vector<int> sizes;
      int i=0;
      int t=0;
      for(int j=0; j<indicators.size(); j++){
	PTENS_ASSRT(indicators[j]>=i);
	if(indicators [j]>i){
	  sizes.push_back(j-t);
	  t=j;
	  i=indicators[j];
	}
      }
      sizes.push_back(indicators.size()-t);
      return BatchedGgraph::from_edge_list(sizes,M);
    }



  public: // ---- Access --------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    int getn() const{
      int t=0;
      for(auto& p:obj->obj)
	t+=p->getn();
      return t;
    }

    Ggraph operator[](const int i) const{
      PTENS_ASSRT(i<size());
      return obj->obj[i];
    }
    

  public: // ---- Operations ----------------------------------------------------------------------------------


    BatchedGgraph permute(const cnine::permutation& pi) const{
      return BatchedGgraph(new OBJ(obj->permute(pi)));
    }

    BatchedAtomsPackBase subgraphs(const Subgraph& H) const{
      return obj->subgraphs(H.obj);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "ptens::BatchedGgraph";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedGgraph& x){
      stream<<x.str(); return stream;}

  };

}

#endif 
