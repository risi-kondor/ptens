#ifndef _ptens_AtomsPack
#define _ptens_AtomsPack

#include <map>

#include "IntTensor.hpp"
#include "Atoms.hpp"


namespace ptens{

  class AtomsPack: public cnine::IntTensor{
  public:

    typedef cnine::IntTensor itensor;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack(const int   _k, const int _dev=0):
      itensor({0,_k},cnine::fill_zero(),_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    AtomsPack(const int _n, const int _k, const FILLTYPE& dummy, const int _dev=0):
      itensor({_n,_k},dummy,_dev){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    static AtomsPack raw(const int _n, const int _k, const int _dev=0){
      return AtomsPack(_n,_k,cnine::fill_raw(),_dev);}

    static AtomsPack zero(const int _n, const int _k, const int _dev=0){
      return AtomsPack(_n,_k,cnine::fill_zero(),_dev);}

    static AtomsPack sequential(const int _n, const int _k, const int _dev=0){
      return AtomsPack(_n,_k,cnine::fill_sequential(),_dev);}


  public: // ---- Access -------------------------------------------------------------------------------------


    int getk() const{
      return dim(1);
    }

    int getn() const{
      return dim(0);
    }

    vector<int> operator()(const int i) const{
      int k=getk();
      vector<int> R(k);
      for(int j=0; j<k; j++)
	R[j]=get(i,j);
      return R;
    }

    int operator()(const int i, const int j) const{
      for(int a=0; a<dims[1] ;a++)
	if(get(i,a)==j) return a;
      return -1;
    }

    vector<int> operator()(const int i, const vector<int>& I) const{
      const int k=I.size();
      vector<int> r(k);
      for(int i=0; i<k; i++)
	r[i]=(*this)(i,I[i]);
      return r;
    }

    bool includes(const int i, const int j) const{
      for(int a=0; a<dims[1] ;a++)
	if(get(i,a)==j) return true;
      return false;
    }

    void push_back(const vector<int>& v){
      assert(v.size()==getk());
      push_back_slice0(IntTensor(v));
    }

    void push_back(const IntTensor& v){
      assert(v.getk()==1);
      assert(v.dim(0)==getk());
      push_back_slice0(v);
    }

    
  public: // ---- Operations ---------------------------------------------------------------------------------


    vector<int> intersect(const int i, const vector<int>& I) const{
      vector<int> r;
      for(auto p: I)
	if(includes(i,p)) r.push_back(p);
      return r;
    }


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
