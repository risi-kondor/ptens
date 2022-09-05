#ifndef _PtensorSubpackSpecializer
#define _PtensorSubpackSpecializer

#include "Cgraph.hpp"
#include "iipair.hpp"


namespace ptens{

  class PtensorSubgraphSpecializer{
  public:

    unordered_map<iipair,Cgraph*> graphs;

    ~PtensorSubgraphSpecializer(){
      for(auto p:graphs)
	delete p.second;
    }

    Cgraph& graph(const int i, const int j){
      auto it=graphs.find(iipair(i,j));
      if(it==graphs.end()){
	Cgraph* G=new Cgraph();
	graphs[iipair(i,j)]=G;
	return *G;
      }
      return *it->second;
    }

    void forall(const std::function<void(const int, const int, Cgraph&)>& lambda){
      for(auto p: graphs)
	lambda(p.first.i0,p.first.i1,*p.second);
    }

  };

}

#endif 
