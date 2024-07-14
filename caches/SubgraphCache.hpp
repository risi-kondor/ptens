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

#ifndef _Ptens_SubgraphCache
#define _Ptens_SubgraphCache

#include <unordered_set>

#include "SubgraphObj.hpp"


namespace ptens{


  template<typename TYPE>
  class shared_ptr_wrapper{
  public:

    shared_ptr<TYPE> obj;

    shared_ptr_wrapper(const shared_ptr<TYPE>& x):
      obj(x){}

    bool operator==(const shared_ptr_wrapper& x) const{
      return (*obj)==(*x.obj);
    }

  };

}


namespace std{
  
  template<typename TYPE>
  struct hash<ptens::shared_ptr_wrapper<TYPE> >{
  public:
    size_t operator()(const ptens::shared_ptr_wrapper<TYPE>& x) const{
      return hash<TYPE>()(*x.obj);
    }
  };
}



namespace ptens{

  class SubgraphCache: public std::unordered_set<shared_ptr_wrapper<SubgraphObj> >{
  public:

    typedef std::unordered_set<shared_ptr_wrapper<SubgraphObj> > BASE;

    template<typename TYPE>
    shared_ptr<SubgraphObj> emplace(const TYPE& x){
      auto r=make_shared<SubgraphObj>(x);
      this->BASE::insert(r);
      return r;
    }

    template<typename ARG0, typename ARG1>
    shared_ptr<SubgraphObj> emplace(const ARG0& x, const ARG1& y){
      auto r=make_shared<SubgraphObj>(x,y);
      BASE::insert(r);
      return r;
    }

    template<typename ARG0, typename ARG1, typename ARG2>
    shared_ptr<SubgraphObj> emplace(const ARG0& x, const ARG1& y, const ARG2& z){
      auto r=make_shared<SubgraphObj>(x,y,z);
      BASE::insert(r);
      return r;
    }

    shared_ptr<SubgraphObj> insert(const SubgraphObj& x){
      auto r=make_shared<SubgraphObj>(x);
      BASE::insert(r);
      return r;
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:*this)
	oss<<*p.obj<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SubgraphCache& x){
      stream<<x.str(); return stream;}
    

    
  };

}

#endif 
