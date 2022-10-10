#ifndef _ptens_AtomsPack
#define _ptens_AtomsPack

#include <map>

#include "array_pool.hpp"
#include "Atoms.hpp"


namespace ptens{

  class AtomsPack: public cnine::array_pool<int>{
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

    static AtomsPack random(const int n, const float p=0.5){
      AtomsPack R;
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<n; i++){
	vector<int> v;
	for(int j=0; j<n; j++)
	  if(distr(rndGen)<p)
	    v.push_back(j);
	R.push_back(v);
      }
      return R;
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


  public: // ---- Copying ------------------------------------------------------------------------------------


    AtomsPack(const AtomsPack& x):
      array_pool(x){
      PTENS_COPY_WARNING();
      //cout<<"AtomsPack copied"<<endl;
    }

    AtomsPack(AtomsPack&& x):
      array_pool(std::move(x)){
      PTENS_MOVE_WARNING();
      //cout<<"AtomsPack moved"<<endl;
    }

    AtomsPack& operator=(const AtomsPack& x){
      PTENS_ASSIGN_WARNING();
      cnine::array_pool<int>::operator=(x);
      return *this;
    }



  public: // ---- Views --------------------------------------------------------------------------------------


    AtomsPack(cnine::array_pool<int>&& x):
      cnine::array_pool<int>(std::move(x)){}


  public: // ---- Views --------------------------------------------------------------------------------------


    AtomsPack view(){
      return AtomsPack(cnine::array_pool<int>::view());
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    Atoms operator[](const int i) const{
      return Atoms(cnine::array_pool<int>::operator()(i));
    }

    int tsize0() const{
      return size();
    }

    int tsize1() const{
      int t=0;
      for(int i=0; i<size(); i++)
	t+=size_of(i);
      return t;
    }

    int tsize2() const{
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


    string classname() const{
      return "AtomsPack";
    }


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
