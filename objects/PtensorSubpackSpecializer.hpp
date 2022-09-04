#ifndef _PtensorSubpackSpecializer
#define _PtensorSubpackSpecializer

#include "Cgraph.hpp"


namespace ptens{

  class PSSiiPair{
  public:

    int i0;
    int i1;
    
    PSSiiPair(const int _i0, const int _i1): i0(_i0), i1(_i1){}

    bool operator==(const PSSiiPair& x) const{
      return (i0==x.i0)&&(i1==x.i1);
    }

  };

}


namespace std{
  template<>
  struct hash<ptens::PSSiiPair>{
  public:
    size_t operator()(const ptens::PSSiiPair& sgntr) const{
      return (hash<int>()(sgntr.i0)<<1)^hash<int>()(sgntr.i1); 
    }
  };
}



namespace ptens{

  class PtensorSubgraphSpecializer{
  public:

    unordered_map<PSSiiPair,Cgraph*> graphs;

    ~PtensorSubgraphSpecializer(){
      for(auto p:graphs)
	delete p.second;
    }

    Cgraph& graph(const int i, const int j){
      auto it=graphs.find(PSSiiPair(i,j));
      if(it==graphs.end()){
	Cgraph* G=new Cgraph();
	graphs[PSSiiPair(i,j)]=G;
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
