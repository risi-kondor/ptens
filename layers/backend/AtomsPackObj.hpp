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


#ifndef _ptens_AtomsPackObj
#define _ptens_AtomsPackObj

#include <map>

#include "observable.hpp"
#include "ptr_indexed_object_bank.hpp"
#include "ptr_arg_indexed_object_bank.hpp"
#include "plist_indexed_object_bank.hpp"
#include "array_pool.hpp"
#include "labeled_forest.hpp"
#include "cpermutation.hpp"
#include "map_of_lists.hpp"
#include "once.hpp"
//#include "monitored.hpp"
#include "Ltensor.hpp"

#include "Atoms.hpp"


namespace ptens{

  class AtomsPackTag0;
  class AtomsPackTag1;
  class AtomsPackTag2;

  class AtomsPackTagObj0;
  class AtomsPackTagObj1;
  class AtomsPackTagObj2;

  class GatherPlanObj;


  class AtomsPackObj: public cnine::array_pool<int>, public cnine::observable<AtomsPackObj>{
  public:

    typedef cnine::array_pool<int> BASE;
    using  BASE::BASE;

    int constk=0;
    int order=0;
    mutable int _tsize2=-1;
    mutable vector<int> offsets2;

    mutable shared_ptr<AtomsPackTagObj0> cached_tag0;
    mutable shared_ptr<AtomsPackTagObj1> cached_tag1;
    mutable shared_ptr<AtomsPackTagObj2> cached_tag2;

    //mutable vector<weak_ptr<LayerMapObj> > related_layermaps;
    mutable vector<weak_ptr<GatherPlanObj> > related_gatherplans;


    ~AtomsPackObj(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackObj():
      observable(this){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackObj(const int n, const int k, const cnine::fill_raw& dummy):
      BASE(n,k),
      observable(this){
      constk=k;
    }

    // needed by SubgrapgListCache
    AtomsPackObj(const int N):
      AtomsPackObj(){
      for(int i=0; i<N; i++)
	push_back(vector<int>({i}));
      //cout<<"Make AtomsPackObj 1"<<endl;
    }

    // eliminate?
    AtomsPackObj(const int N, const int k):
      AtomsPackObj(){
      vector<int> v;
      for(int j=0; j<k; j++) 
	  v.push_back(j);
      for(int i=0; i<N; i++){
	push_back(v);
      }
    }

    AtomsPackObj(const vector<vector<int> >& x):
      AtomsPackObj(){
      for(auto& p:x)
	push_back(p);
    }

    AtomsPackObj(const initializer_list<initializer_list<int> >& x):
      AtomsPackObj(){
      for(auto& p:x)
	push_back(p);

    }

    AtomsPackObj(const cnine::labeled_forest<int>& forest):
      AtomsPackObj(){
      for(auto p:forest)
	p->for_each_maximal_path([&](const vector<int>& x){
	    push_back(x);});
    }


  public: // ---- Static Constructors ------------------------------------------------------------------------


    static AtomsPackObj random(const int n, const int m, const float p=0.5){
      AtomsPackObj R;
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<n; i++){
	vector<int> v;
	for(int j=0; j<m; j++)
	  if(distr(rndGen)<p)
	    v.push_back(j);
	R.push_back(v);
      }
      return R;
    }

   static AtomsPackObj random0(const int n){
      AtomsPackObj R;
      uniform_int_distribution<> distr(0,n-1);
      for(int i=0; i<n; i++){
	R.push_back({distr(rndGen)});
      }
      return R;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    AtomsPackObj(const AtomsPackObj& x):
      BASE(x),
      observable(this),
      _tsize2(x._tsize2),
      offsets2(x.offsets2),
      cached_tag0(x.cached_tag0),
      cached_tag1(x.cached_tag1),
      cached_tag2(x.cached_tag2){
      PTENS_COPY_WARNING();
      constk=x.constk;
      order=x.order;
    }

    AtomsPackObj(AtomsPackObj&& x): 
      array_pool(std::move(x)),
      observable(this),
      _tsize2(x._tsize2),
      offsets2(std::move(x.offsets2)),
      cached_tag0(x.cached_tag0),
      cached_tag1(x.cached_tag1),
      cached_tag2(x.cached_tag2){
      PTENS_MOVE_WARNING();
      constk=x.constk;
      order=x.order;
    }

    AtomsPackObj& operator=(const AtomsPackObj& x){
      PTENS_ASSIGN_WARNING();
      //cout<<"AtomsPackAssigned!"<<endl;
      cnine::array_pool<int>::operator=(x);
      constk=x.constk;
      return *this;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    AtomsPackObj(cnine::array_pool<int>&& x):
      cnine::array_pool<int>(std::move(x)),
      observable(this){}

    AtomsPackObj(const cnine::Tensor<int>& M):
      AtomsPackObj(cnine::array_pool<int>(M)){
      PTENS_ASSRT(M.ndims()==2);
      constk=M.dim(1);
    }


  public: // ---- Views --------------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int getk() const{
      return order;
    }

    Atoms operator[](const int i) const{
      return Atoms(cnine::array_pool<int>::operator()(i));
    }

    int n_intersects(const AtomsPackObj& y, const int i, const int j) const{
      int k=0;
      int ni=size_of(i);
      int nj=y.size_of(j);
      for(int a=0; a<ni; a++){
	int ix=(*this)(i,a);
	for(int b=0; b<nj; b++)
	  if(y(j,b)==ix){
	    k++;
	    break;
	  }
      }
      return k;
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

    //void release_cached_packs(){
    //cached_pack0.reset();
    //cached_pack1.reset();
    //cached_pack2.reset();
    //cache_packs=false;
    //}


  public: // ---- 0th order layout -----------------------------------------------------------------------------------


    int tsize0() const{
      return size();
    }

    int nrows0() const{
      return tsize0();
    }

    int nrows0(const int i) const{
      PTENS_ASSRT(i<size());
      return 1;
    }

    int row_offset0(const int i) const{
      PTENS_ASSRT(i<size());
      return i;
    }

    int offset0(const int i) const{
      return i;
    }

    int index_of0(const int i) const{
      return i;
    }

  public: // ---- 1st order layout -----------------------------------------------------------------------------------


    int tsize1() const{
      return BASE::get_tail(); 
    }

    int nrows1() const{
      return tsize1();
    }

    int nrows1(const int i) const{
      PTENS_ASSRT(i<size());
      return BASE::size_of(i);
    }

    int row_offset1(const int i) const{
      PTENS_ASSRT(i<size());
      return BASE::offset(i);
    }

    int offset1(const int i) const{
      return BASE::offset(i);
    }

    int index_of1(const int i, const int j0) const{
      return BASE::offset(i)+j0;
    }


  public: // ---- 2nd order layout -----------------------------------------------------------------------------------


    int tsize2() const{
      if(_tsize2==-1){
	int t=0; 
	for(int i=0; i<size(); i++) 
	  t+=pow(BASE::size_of(i),2);
	_tsize2=t;
      }
      return _tsize2;
    }

    int nrows2() const{
      return tsize2();
    }
    
    int nrows2(const int i) const{
      PTENS_ASSRT(i<size());
      return pow(BASE::size_of(i),2);
    }

    int row_offset2(const int i) const{
      PTENS_ASSRT(i<size());
      if(offsets2.size()==0){
	offsets2.resize(size());
	int t=0;
	for(int i=0; i<size(); i++){
	  offsets2[i]=t;
	  t+=pow(BASE::size_of(i),2);
	}
      }
      return offsets2[i];
    }

    int offset2(const int i) const{
      return row_offset2(i);
    }

    int index_of(const int i, const int j0, const int j1) const{
      return row_offset2(i)+j0*size_of(i)+j1;
    }


  public: // ---- Concatenation ------------------------------------------------------------------------------


    /*
    static shared_ptr<AtomsPackObj> cat(const vector<shared_ptr<AtomsPackObj> >& list){
      PTENS_ASSRT(list.size()>0);
      vector<AtomsPackObj*> v;
      bool first=true;
      for(auto p:list)
	if(first) first=false; 
	else v.push_back(p.get());
      return (*list.begin())->cat_maps(cnine::plist<AtomsPackObj*>(v));
    }

    static shared_ptr<AtomsPackObj> cat(const vector<reference_wrapper<AtomsPackObj> >& list){
      PTENS_ASSRT(list.size()>0);
      vector<AtomsPackObj*> v;
      bool first=true;
      for(auto p:list)
	if(first) first=false; 
	else v.push_back(&p.get());
      return list.begin()->get().cat_maps(cnine::plist<AtomsPackObj*>(v));
    }

    AtomsPackObj cat_with(const vector<AtomsPackObj*> list){
      vector<reference_wrapper<cnine::array_pool<int> > > v;
      v.push_back(*this);
      for(auto p:list)
	v.push_back(*p);
      return AtomsPackObj(cnine::array_pool<int>::cat(v));
    }
    */

    static AtomsPackObj cat(const vector<shared_ptr<AtomsPackObj> > list){
      vector<reference_wrapper<cnine::array_pool<int> > > v;
      for(auto p:list)
	v.push_back(*p);
      return AtomsPackObj(cnine::array_pool<int>::cat(v));
    }

    static AtomsPackObj cat(const vector<AtomsPackObj*> list){
      vector<reference_wrapper<cnine::array_pool<int> > > v;
      for(auto p:list)
	v.push_back(*p);
      return AtomsPackObj(cnine::array_pool<int>::cat(v));
    }



  public: // ---- to_nodes_map -------------------------------------------------------------------------------

    /*
    cnine::oncep<cnine::GatherMap> gather_to_nodes_map=
      cnine::oncep<cnine::GatherMap>([&](){

	cnine::map_of_lists<int,int> lists;
	for(int i=0; i<size(); i++)
	  for(int j=0; j<size_of(i); j++)
	    lists[(*this)(i,j)].push_back(i);
	int n_dest=lists.size();

	cnine::GatherMap* R=new cnine::GatherMap(n_dest,get_tail(),cnine::fill_raw());

	int i=0;
	int _tail=3*n_dest;
	for(auto& p:lists){
	  vector<int>& list=p.second;
	  const int n=list.size();

	  R->arr[3*i]=_tail;
	  R->arr[3*i+1]=n;
	  R->arr[3*i+2]=p.first;

	  for(int j=0; j<n; j++){
	    R->arr[_tail+2*j]=list[j];
	    *reinterpret_cast<float*>(R->arr+_tail+2*j+1)=1.0;
	  }

	  i++;
	  _tail+=2*n;
	}

	R->to_device(1);
	return R;
      });
    */

  public: // ---- Operations ---------------------------------------------------------------------------------


    AtomsPackObj permute(const cnine::permutation& pi){
      PTENS_ASSRT(get_dev()==0);
      array_pool<int> A;
      for(int i=0; i<size(); i++){
	vector<int> v=(*this)(i);
	int len=v.size();
	vector<int> w(len);
	for(int j=0; j<len; j++)
	  w[j]=pi(v[j]);
	A.push_back(w);
      }
      return A;
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPackObj";
    }

    string repr() const{
      return "<AtomsPackObj n="+to_string(size())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const AtomsPackObj& v){
      stream<<v.str(); return stream;}}
;

  

  class AtomsPackObj0: public AtomsPackObj{
  public:

    int nrows() const{
      return tsize0();
    }

    int nrows(const int i) const{
      return 1;
    }

    int row_offset(const int i) const{
      return i;
    }

    int offset(const int i) const{
      return i;
    }

    int index_of(const int i) const{
      return i;
    }

  };


  class AtomsPackObj1: public AtomsPackObj{
  public:

    typedef cnine::array_pool<int> BASE;

    int nrows() const{
      return tsize1();
    }

    int nrows(const int i) const{
      return BASE::size_of(i);
    }

    int row_offset(const int i) const{
      return BASE::offset(i);
    }

    int offset(const int i) const{
      return BASE::offset(i);
    }

    int index_of(const int i, const int j0) const{
      return BASE::offset(i)+j0;
    }

  };


  class AtomsPackObj2: public AtomsPackObj{
  public:

    typedef cnine::array_pool<int> BASE;

    int nrows() const{
      return tsize2();
    }
    
    int nrows(const int i) const{
      return pow(BASE::size_of(i),2);
    }

    int row_offset(const int i) const{
      return row_offset2(i);
    }

    int offset(const int i) const{
      return row_offset2(i);
    }

    int index_of(const int i, const int j0, const int j1) const{
      return row_offset2(i)+j0*size_of(i)+j1;
    }

  };


}


#endif 

 
