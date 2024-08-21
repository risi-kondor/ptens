/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 */

#ifndef _ptens_atoms
#define _ptens_atoms

#include <map>


namespace ptens{

  class Atoms: public vector<int>{
  public:

    map<int,int> lookup;


  public: // ---- Constructors -------------------------------------------------------------------------------


    Atoms(){}

    Atoms(const initializer_list<int>& list){
      for(auto p:list)
	push_back(p);
      for(int i=0; i<size(); i++)
	lookup[(*this)[i]]=i;
    }

    Atoms(const vector<int>& x):
      vector<int>(x){
      for(int i=0; i<size(); i++)
	lookup[(*this)[i]]=i;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    static Atoms sequential(const int _k){
      vector<int> v(_k);
      for(int i=0; i<_k; i++) v[i]=i;
      return Atoms(v);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int operator()(const int i) const{
      auto it=lookup.find(i);
      if(it==lookup.end()) return 0;
      return it->second;
    }


    vector<int> operator()(const vector<int>& I) const{
      const int k=I.size();
      vector<int> r(k);
      for(int i=0; i<k; i++)
	r[i]=(*this)(I[i]);
      return r;
    }


    void push_back(const int i){
      lookup[i]=size();
      vector<int>::push_back(i);
    }


    bool includes(const int i) const{
      return lookup.find(i)!=lookup.end();
    }


    void foreach(const std::function<void(const int)>& lambda) const{
      for(int i=0; i<size(); i++)
	lambda((*this)[i]);
    }

    
  public: // ---- Operations ---------------------------------------------------------------------------------


    Atoms intersect(const Atoms& y) const{
      Atoms R;
      for(auto p: *this)
	if(y.includes(p)) R.push_back(p);
      return R;
    }

    std::pair<vector<int>,vector<int> > intersecting(const Atoms& y) const{
      vector<int> xi;
      vector<int> yi;
      for(int i=0; i<size(); i++){
	auto it=y.lookup.find((*this)[i]);
	if(it!=y.lookup.end()){
	  xi.push_back(i);
	  yi.push_back(it->second);
	}
      }
      return make_pair(xi,yi);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<"[";
      int k=size(); //why?
      for(int i=0; i<k-1; i++)
	oss<<(*this)[i]<<",";
      if(size()>0) oss<<(*this)[size()-1];
      oss<<"]";
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const Atoms& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
