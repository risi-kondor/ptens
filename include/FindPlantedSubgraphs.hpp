#ifndef _Hgraph
#define _Hgraph

//#include <set>
#include "Ptens_base.hpp"
#include "labeled_tree.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Hgraph.hpp"


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

      for(int i=0; i<G.getn(); i++){
	labeled_tree* T=new labeled_tree(i);
	matches.push_back(T);
	if(!make_subtree(*T,0)){
	  delete T;
	  matches.pop_back();
	}
      }
    }


    operator AindexPack(){
      AindexPack R;
      for(auto p:matches)
	p->forall_maximal_paths([&](const vector<int>& x){
	    R.push_back(x);});
      return R;
    }


  private:


    bool make_subtree(labeled_tree& node, const int m){

      PTENS_ASSRT(m<traversal.size());
      const int v=traversal[m].first;
      const int w=node.label;

      for(auto& p:H.row(v)){
	if(assignment[p.first]==-1) continue;
	if(p.second!=G(w,assignment[p.first])) return false;
      }
      for(auto& p:G.row(w)){
	auto it=std::find(assignment.begin(),assignment.end(),p.first);
	if(it==assignment.end()) continue;
	if(p.second!=H(v,traversal[it-assignment.begin()])) return false;
      }

      assignment[v]=w;
      if(m==n-1) return !T.contains_some_permutation_of(assignment);
      
      const int newparent=assignment(traversal[traversal[m+1].second]);
      // try to match next vertex in traversal to each neighbor of newparent  
      for(auto& w:G.neigbors(newparent)){
	if(std::find(assignment.begin(),assignment.end(),w)!=assignment.end()) continue;
	labeled_tree* T=new labeled_tree(w);
	node.push_back(T);
	if(!make_subtree(*T,m+1)){
	  delete T;
	  node.children.pop_back();
	}
      }

      return node.children.size()>0;
    }

  };

}

#endif
