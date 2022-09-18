#ifndef _Hgraph
#define _Hgraph

#include <set>
#include "Ptens_base.hpp"
#include "SparseMx.hpp"
#include "AtomsPack.hpp"


namespace ptens{

  class Hgraph: public SparseMx{
  public:

    using SparseMx::SparseMx;

    //Hgraph(const SparseMx& M):
    //SparseMx(M){
    //}

    mutable vector<AtomsPack*> _nhoods; 

    ~Hgraph(){
      for(auto p:_nhoods)
	delete p;
    }


  public:

    Hgraph(const int _n):
      Hgraph(_n,_n){}


    static Hgraph random(const int _n, const float p=0.5){
      return SparseMx::random_symmetric(_n,p);
    }

    Hgraph(const SparseMx& x):
      SparseMx(x){}


  public:

    AtomsPack nhoods(const int i) const{
      if(_nhoods.size()==0) _nhoods.push_back(new AtomsPack(n));
      for(int j=_nhoods.size(); j<=i; j++){
	const AtomsPack& prev=*_nhoods.back(); 
	AtomsPack* newlevel=new AtomsPack();
	for(int i=0; i<prev.size(); i++){
	  vector<int> v=prev(i);
	  std::set<int> w; //=new std::set<int>;
	  for(auto p:v){
	    w.insert(p);
	    for(auto q: *const_cast<Hgraph&>(*this).lists[p])
	      w.insert(q.first);
	  }
	  newlevel->push_back(w);
	}
	_nhoods.push_back(newlevel);
      }
      
      return AtomsPack(*_nhoods[i]);
    }


  };

}

#endif
