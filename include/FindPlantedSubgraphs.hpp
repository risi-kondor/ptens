#ifndef _FindPlantedSubgraphs
#define _FindPlantedSubgraphs

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

      for(auto p:Htraversal) cout<<p.first<<" "; cout<<endl;
      for(auto p:Htraversal) cout<<p.second<<" "; cout<<endl;

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
      int i=0;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){
	    R.push_back(i++,x);});
      return R;
    }


  private:


    bool make_subtree(labeled_tree& node, const int m){

      PTENS_ASSRT(m<Htraversal.size());
      const int v=Htraversal[m].first;
      const int w=node.label;
      //cout<<"trying "<<v<<" against "<<w<<" at level "<<m<<endl;

      for(auto& p:H.row(v)){
	if(assignment[p.first]==-1) continue;
	if(p.second!=G(w,assignment[p.first])) return false;
      }
      for(auto& p:G.row(w)){
	auto it=std::find(assignment.begin(),assignment.end(),p.first);
	if(it==assignment.end()) continue;
	if(p.second!=H(v,Htraversal[it-assignment.begin()].first)) return false;
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

}

#endif
