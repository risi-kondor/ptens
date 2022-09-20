#ifndef _ptens_AtomsPack
#define _ptens_AtomsPack

#include <map>

#include "array_pool.hpp"
#include "Atoms.hpp"
//#include "GraphNhoods.hpp"


namespace ptens{

  class AtomsPack: public array_pool<int>{
  public:


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack(){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack(const int N){
      for(int i=0; i<N; i++)
	push_back(vector<int>({i}));
    }

    AtomsPack(const int N, const int k){
      for(int i=0; i<N; i++){
	vector<int> v;
	for(int j=0; j<k; j++) 
	  v.push_back(j);
	push_back(v);
      }
    }

    AtomsPack(const vector<vector<int> >& x){
      for(auto& p:x)
	push_back(p);
    }

    /*
    AtomsPack(const Nhoods& N){
      for(int i=0; i<N.nhoods.size(); i++){
	vector<int> v;
	for(int q: *N.nhoods[i])
	  v.push_back(q);
	push_back(v);
      }
    }
    */


  public: // ---- Access -------------------------------------------------------------------------------------


    Atoms operator[](const int i) const{
      return Atoms(array_pool<int>::operator()(i));
    }

    int tsize0(){
      return size();
    }

    int tsize1(){
      int t=0;
      for(int i=0; i<size(); i++)
	t+=sizeof(i);
      return t;
    }

    int tsize2(){
      int t=0;
      for(int i=0; i<size(); i++)
	t+=size_of(i)*size_of(i);
      return t;
    }

    
  public: // ---- Operations ---------------------------------------------------------------------------------

    /*
    vector<int> intersect(const int i, const vector<int>& I) const{
      vector<int> r;
      for(auto p: I)
	if(includes(i,p)) r.push_back(p);
      return r;
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    string str(const string indent="") const{
      ostringstream oss;
      fo
      oss<<"(";
      for(int i=0; i<size()-1; i++)
	oss<<(*this)[i]<<",";
      if(size()>0) oss<<(*this)[size()-1];
      oss<<")";
      return oss.str();
    }
    */


    friend ostream& operator<<(ostream& stream, const AtomsPack& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
