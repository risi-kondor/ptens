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

#ifndef _ptens_AtomsPack
#define _ptens_AtomsPack

#include <map>

#include "array_pool.hpp"
#include "labeled_forest.hpp"
#include "Atoms.hpp"


namespace ptens{

  class AtomsPack: public cnine::array_pool<int>{
  public:

    //int k=-1;
    typedef cnine::array_pool<int> BASE;
    using  BASE::BASE;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack(){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack(const int N){
      //k=1;
      for(int i=0; i<N; i++)
	push_back(vector<int>({i}));
    }

    AtomsPack(const int N, const int k){
      //k=_k;
      vector<int> v;
      for(int j=0; j<k; j++) 
	  v.push_back(j);
      for(int i=0; i<N; i++){
	push_back(v);
      }
    }

    AtomsPack(const vector<vector<int> >& x){
      for(auto& p:x)
	push_back(p);
    }

    AtomsPack(const initializer_list<initializer_list<int> >& x){
      for(auto& p:x)
	push_back(p);
    }

    AtomsPack(const cnine::labeled_forest<int>& forest){
      for(auto p:forest)
	p->for_each_maximal_path([&](const vector<int>& x){
	    push_back(x);});
    }


  public: // ---- Static Constructors ------------------------------------------------------------------------


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
      array_pool(x)/*, k(x.k)*/{
      PTENS_COPY_WARNING();
    }

    AtomsPack(AtomsPack&& x):
      array_pool(std::move(x))/*, k(x.k)*/{
      PTENS_MOVE_WARNING();
    }

    AtomsPack& operator=(const AtomsPack& x){
      PTENS_ASSIGN_WARNING();
      cnine::array_pool<int>::operator=(x);
      /*k=x.k;*/
      return *this;
    }



  public: // ---- Conversions --------------------------------------------------------------------------------


    AtomsPack(cnine::array_pool<int>&& x/*, const int _k=-1*/):
      cnine::array_pool<int>(std::move(x))/*, k(_k)*/{}

    AtomsPack(const cnine::Tensor<int>& M/*, const int _k=-1*/):
      cnine::array_pool<int>(M)/*, k(_k)*/{}


  public: // ---- Views --------------------------------------------------------------------------------------


    AtomsPack view(){
      //return AtomsPack(cnine::array_pool<int>::view(),-1); // hack
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

    cnine::array_pool<int> dims1(const int nc) const{
      array_pool<int> R;
      for(int i=0; i<size(); i++)
	R.push_back({size_of(i),nc});
      return R;
    }

    cnine::array_pool<int> dims2(const int nc) const{
      array_pool<int> R;
      for(int i=0; i<size(); i++)
	R.push_back({size_of(i),size_of(i),nc});
      return R;
    }


  public: // ---- Concatenation ------------------------------------------------------------------------------


    static AtomsPack cat(const vector<reference_wrapper<AtomsPack> >& list){
      return AtomsPack(cnine::array_pool<int>::cat
	(cnine::mapcar<reference_wrapper<AtomsPack>,reference_wrapper<array_pool<int> > >
	  (list,[](const reference_wrapper<AtomsPack>& x){
	    return reference_wrapper<array_pool<int> >(x.get());})));
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
