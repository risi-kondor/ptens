#ifndef _Hgraph
#define _Hgraph

#include <set>
#include "Ptens_base.hpp"
#include "SparseRmatrix.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "GatherMap.hpp"


namespace ptens{


  class Hgraph: public cnine::SparseRmatrix{
  public:


    using cnine::SparseRmatrix::SparseRmatrix;

    mutable Hgraph* _reverse=nullptr;
    mutable cnine::CSRmatrix<float>* gmap=nullptr; 
    mutable cnine::GatherMap* bmap=nullptr;
    mutable vector<AtomsPack*> _nhoods; 

    ~Hgraph(){
      // if(_reverse) delete _reverse; // hack!
      for(auto p:_nhoods)
	delete p;
      if(gmap) delete gmap;
      if(bmap) delete bmap;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    Hgraph(const int _n):
      Hgraph(_n,_n){}


    static Hgraph random(const int _n, const float p=0.5){
      return cnine::SparseRmatrix::random_symmetric(_n,p);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    Hgraph(const Hgraph& x):
      SparseRmatrix(x){}

    Hgraph(Hgraph&& x):
      SparseRmatrix(std::move(x)){}

    Hgraph& operator=(const Hgraph& x)=delete;


  public: // ---- Conversions --------------------------------------------------------------------------------


    Hgraph(const cnine::SparseRmatrix& x):
      cnine::SparseRmatrix(x){}


  public: // ---- Access -------------------------------------------------------------------------------------


    const Hgraph& reverse() const{
      if(!_reverse) _reverse=new Hgraph(transp());
      //if(_reverse) const_cast<Hgraph&>(*this).make_reverse();
      return *_reverse;
    }

    const cnine::CSRmatrix<float>& get_gmap() const{
      if(!gmap) gmap=new cnine::CSRmatrix<float>(CSRmatrix());
      return *gmap;
    }

    const cnine::GatherMap& get_bmap() const{
      if(!bmap) bmap=new cnine::GatherMap(broadcast_map());
      return *bmap;
    }


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


    AtomsPack merge(const AtomsPack& x) const{
      PTENS_ASSRT(m==x.size());
      AtomsPack R;
      for(int i=0; i<n; i++){
	std::set<int> w;
	w.insert(i);
	for(auto q: const_cast<Hgraph&>(*this).row(i)){
	  auto a=x[q.first];
	  for(auto p:a)
	    w.insert(p);
	}
	R.push_back(w);
      }
      return R;
    }


    pair<AindexPack,AindexPack> intersects(const AtomsPack& inputs, const AtomsPack& outputs, const bool self=0) const{
      PTENS_ASSRT(outputs.size()==n);
      PTENS_ASSRT(inputs.size()==m);
      AindexPack in_indices;
      AindexPack out_indices;
      forall_edges([&](const int i, const int j, const float v){
	  Atoms in=inputs[j];
	  Atoms out=outputs[i];
	  Atoms common=out.intersect(in);
	  auto p=out(common);
	  in_indices.push_back(j,in(common));
	  out_indices.push_back(i,out(common));
	}, self);
      out_indices.bmap=&get_bmap();
      return make_pair(in_indices, out_indices);
    }


    cnine::GatherMap broadcast_map() const{
      int nlists=0;
      int nedges=0;
      for(auto q:lists)
	if(q.second->size()>0){
	  nlists++;
	  nedges+=q.second->size();
	}

      cnine::GatherMap R(nlists,nedges);
      int i=0;
      int tail=3*nlists;
      for(auto q:lists){
	int len=q.second->size();
	if(len==0) continue;
	R.arr[3*i]=tail;
	R.arr[3*i+1]=len;
	R.arr[3*i+2]=q.first;
	int j=0;
	for(auto p:*q.second){
	  R.arr[tail+2*j]=p.first;
	  *reinterpret_cast<float*>(R.arr+tail+2*j+1)=p.second;
	  j++;
	}
	tail+=2*len;
	i++;
      }
      return R;
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "Hgraph";
    }



  };

}

#endif
