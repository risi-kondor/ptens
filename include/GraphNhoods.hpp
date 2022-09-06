#ifndef _GraphNhoods
#define _GraphNhoods

#include "Cgraph.hpp"


namespace ptens{

  class Nhoods{
  public:

    vector<set<int>* > nhoods;

    ~Nhoods(){
      for(auto p:nhoods)
	delete p;
    }

  public:

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<nhoods.size(); i++){
	oss<<indent<<i<<": ";
	for(int p:*nhoods[i])
	  oss<<p<<" ";
	oss<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Nhoods& x){
      stream<<x.str(); return stream;}

  };


  class GraphNhoods{
  public:

    vector<Nhoods*> levels;

    ~GraphNhoods(){
      for(auto p:levels)
	delete p;
    }


  public:

    GraphNhoods(const Cgraph& G, const int nlevels){

      Nhoods* level0=new Nhoods();

      /*
      int i=0;
      for(auto p:G.lists){
	int _i=p.first;
	for(int j=i; j<=_i; j++)
	  level0->nhoods.push_back(new set<int>());
	set<int>& nhood=*level0->nhoods.back();
      }
      levels.push_back(level0);
      */

      for(int i=0; i<G.maxi; i++){
	set<int>* nhood=new set<int>;
	nhood->insert(i);
	level0->nhoods.push_back(nhood);
      }
      levels.push_back(level0);

      for(int l=1; l<nlevels; l++){
	Nhoods& prev=*levels.back();
	Nhoods* newlevel=new Nhoods();
	for(int i=0; i<prev.nhoods.size(); i++){
	  set<int>* nhood=new set<int>;
	  for(auto p:*prev.nhoods[i]){
	    nhood->insert(p);
	    for(auto q: *const_cast<Cgraph&>(G).lists[p])
	      nhood->insert(q);
	  }
	  newlevel->nhoods.push_back(nhood);
	}
	levels.push_back(newlevel);
      }

    }


  public:

    Nhoods& level(const int i){
      assert(i<levels.size());
      return *levels[i];
    }

    
  };

}

#endif
