#ifndef _Hgraph
#define _Hgraph

#include <set>
#include "Ptens_base.hpp"
#include "SparseMx.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"


namespace ptens{

  class Hgraph: public SparseMx{
  public:

    using SparseMx::SparseMx;

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


  public: // ---- Access -------------------------------------------------------------------------------------


    void forall_edges(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }


    AtomsPack nhoods(const int i) const{
      if(_nhoods.size()==0) _nhoods.push_back(new AtomsPack(n));
      for(int j=_nhoods.size(); j<=i; j++){
	const AtomsPack& prev=*_nhoods.back();
	assert(prev.size()==n);
	AtomsPack* newlevel=new AtomsPack();
	for(int i=0; i<prev.size(); i++){
	  vector<int> v=prev(i);
	  std::set<int> w;
	  for(auto p:v){
	    w.insert(p);
	    for(auto q: const_cast<Hgraph&>(*this).row(p))
	      w.insert(q.first);
	  }
	  newlevel->push_back(w);
	}
	_nhoods.push_back(newlevel);
      }
      
      return AtomsPack(*_nhoods[i]);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    pair<AindexPack,AindexPack> intersects(const AtomsPack& inputs, const AtomsPack& outputs, const bool self=0) const{
      assert(outputs.size()==n);
      assert(inputs.size()==m);
      AindexPack in_indices;
      AindexPack out_indices;
      forall_edges([&](const int i, const int j, const float v){
	  Atoms in=inputs[j];
	  Atoms out=outputs[i];
	  Atoms common=out.intersect(in);
	  //cout<<j<<"->"<<i<<" "<<in<<" "<<out<<" "<<common<<" "<<Atoms(in(common))<<" "<<Atoms(out(common))<<endl;
	  in_indices.push_back(j,in(common));
	  out_indices.push_back(i,out(common));
	}, self);
      return make_pair(in_indices, out_indices);
    }


  };

}

#endif
