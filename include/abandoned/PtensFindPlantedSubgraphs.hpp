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

#ifndef _FindPlantedSubgraphs
#define _FindPlantedSubgraphs

//#include <set>
#include "Ptens_base.hpp"
#include "labeled_tree.hpp"
#include "labeled_forest.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Hgraph.hpp"
#include "Tensor.hpp"
#include "flog.hpp"


namespace ptens{

  class FindPlantedSubgraphs{
  public:

    typedef Hgraph Graph;
    typedef cnine::labeled_tree<int> labeled_tree;
    typedef cnine::labeled_forest<int> labeled_forest;


    const Graph& G;
    const Graph& H;
    int n;
    vector<pair<int,int> > Htraversal;
    vector<int> assignment;
    labeled_forest matches;


  public:


    FindPlantedSubgraphs(const Graph& _G, const Graph& _H):
      G(_G), H(_H), n(_H.getn()){
      labeled_tree S=H.greedy_spanning_tree();
      Htraversal=S.indexed_depth_first_traversal();
      assignment=vector<int>(n,-1);
      /*cout<<"compute"<<endl;*/

      for(int i=0; i<G.getn(); i++){
	labeled_tree* T=new labeled_tree(i);
	matches.push_back(T);
	if(!make_subtree(*T,0)){
	  delete T;
	  matches.pop_back();
	}
      }
    }

    int nmatches() const{
      int t=0;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){t++;});
      return t;
    }

    operator AindexPack(){
      AindexPack R;
      int i=0;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){
	    R.push_back(i++,x);});
      return R;
    }

    operator cnine::array_pool<int>(){
      cnine::array_pool<int> R;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){
	    R.push_back(x);});
      return R;
    }

    operator cnine::Tensor<int>(){
      int N=nmatches();
      cnine::Tensor<int> R(cnine::Gdims(N,n));
      int t=0;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){
	    for(int i=0; i<n; i++) R.set(t,i,x[i]);
	    t++;});
      return R;
    }


  private:


    bool make_subtree(labeled_tree& node, const int m){

      PTENS_ASSRT(m<Htraversal.size());
      const int v=Htraversal[m].first;
      const int w=node.label;
      //cout<<"trying "<<v<<" against "<<w<<" at level "<<m<<endl;

      if(G.is_labeled && H.is_labeled && (G.labels(w)!=H.labels(v))) return false;

      for(auto& p:H.row(v)){
	if(assignment[p.first]==-1) continue;
	if(p.second!=G(w,assignment[p.first])) return false;
      }
      for(auto& p:G.row(w)){
	auto it=std::find(assignment.begin(),assignment.end(),p.first);
	if(it==assignment.end()) continue;
	if(p.second!=H(v,Htraversal[it-assignment.begin()].first)) return false; // incorrect!!
      }

      assignment[v]=w;
      if(m==n-1){
	node.label=-1;
	bool is_duplicate=matches.contains_rooted_path_consisting_of(assignment);
	node.label=w;
	assignment[v]=-1;
	return !is_duplicate;
      }

      // try to match next vertex in Htraversal to each neighbor of newparent  
      const int newparent=assignment[Htraversal[Htraversal[m+1].second].first];
      for(auto& w:G.neighbors(newparent)){
	if(std::find(assignment.begin(),assignment.end(),w)!=assignment.end()) continue;
	labeled_tree* T=new labeled_tree(w);
	node.push_back(T);
	if(!make_subtree(*T,m+1)){
	  delete T;
	  node.children.pop_back();
	}
      }

      assignment[v]=-1;
      return node.children.size()>0;
    }

  };


  class CachedPlantedSubgraphs{
  public:

    typedef Hgraph Graph;

    cnine::array_pool<int> operator()(const Graph& G, const Graph& H){
      cnine::flog timer("CachedPlantedSubgraphs");
      //if(!G.subgraphlist_cache) G.subgraphlist_cache=new HgraphSubgraphListCache; 
      auto it=G.subgraphlist_cache.find(H);
      if(it!=G.subgraphlist_cache.end()) return *it->second;
      auto newpack=new cnine::array_pool<int>(FindPlantedSubgraphs(G,H));
      G.subgraphlist_cache[H]=newpack;
      return *newpack;
    }
  };


  class CachedPlantedSubgraphsMx{
  public:

    typedef Hgraph Graph;

    shared_ptr<cnine::Tensor<int> > ptr;

    CachedPlantedSubgraphsMx(const Graph& G, const Graph& H){
      cnine::flog timer("CachedPlantedSubgraphsMx");
      //if(!G.subgraphlist_cache) G.subgraphlist_cache=new HgraphSubgraphListCache; 
      auto it=G.subgraphlistmx_cache.find(H);
      if(it!=G.subgraphlistmx_cache.end()) ptr=it->second;
      else{
	shared_ptr<cnine::Tensor<int> > A(new cnine::Tensor<int>(FindPlantedSubgraphs(G,H)));
	G.subgraphlistmx_cache[H]=A;
	ptr=A;
      }
    }

    operator const cnine::Tensor<int>&() const{
      return *ptr;
    }

  };

}

#endif
