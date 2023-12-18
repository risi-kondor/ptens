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
#include "plist_indexed_object_bank.hpp"
#include "array_pool.hpp"
#include "labeled_forest.hpp"
#include "cpermutation.hpp"
#include "map_of_lists.hpp"
#include "once.hpp"

#include "Atoms.hpp"
#include "TransferMap.hpp"


namespace ptens{


  class AtomsPackObj: public cnine::array_pool<int>, public cnine::observable<AtomsPackObj>{
  public:

    typedef cnine::array_pool<int> BASE;
    using  BASE::BASE;


    cnine::ptr_indexed_object_bank<AtomsPackObj,TransferMap> overlap_maps=
      cnine::ptr_indexed_object_bank<AtomsPackObj,TransferMap>([this](const AtomsPackObj& x)
	{return TransferMap(new TransferMapObj<AtomsPackObj>(x,*this));});

    cnine::plist_indexed_object_bank<AtomsPackObj,shared_ptr<AtomsPackObj>> cat_maps=
      cnine::plist_indexed_object_bank<AtomsPackObj,shared_ptr<AtomsPackObj>>([this](const vector<AtomsPackObj*>& v)
	{return shared_ptr<AtomsPackObj>(new AtomsPackObj(cat_with(v)));});

    int constk=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackObj():
      observable(this){}


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackObj(const int n, const int k, const cnine::fill_raw& dummy):
      BASE(n,k),
      observable(this){
      constk=k;
    }

    // eliminate?
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
      //cout<<"Make AtomsPackObj 2"<<endl;

    }

    // eliminate?
    //[[deprecated]]
    //AtomsPackObj(const vector<vector<int> >& x):
    //AtomsPackObj(){
    //for(auto& p:x)
    //push_back(p);
    //cout<<"Make AtomsPackObj 3"<<endl;
    //}

    AtomsPackObj(const initializer_list<initializer_list<int> >& x):
      AtomsPackObj(){
      for(auto& p:x)
	push_back(p);
      cout<<"Make AtomsPackObj 4"<<endl;

    }

    AtomsPackObj(const cnine::labeled_forest<int>& forest):
      AtomsPackObj(){
      for(auto p:forest)
	p->for_each_maximal_path([&](const vector<int>& x){
	    push_back(x);});
      cout<<"Make AtomsPackObj 5"<<endl;
    }


  public: // ---- Static Constructors ------------------------------------------------------------------------


    static AtomsPackObj random(const int n, const float p=0.5){
      AtomsPackObj R;
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


  public: // ---- Copying ------------------------------------------------------------------------------------


    AtomsPackObj(const AtomsPackObj& x):
      array_pool(x),
      observable(this){
      //cout<<"AtomsPackCopied!"<<endl;
      PTENS_COPY_WARNING();
      constk=x.constk;
    }

    AtomsPackObj(AtomsPackObj&& x):
      array_pool(std::move(x)),
      observable(this){
      PTENS_MOVE_WARNING();
      //cout<<"AtomsPackMoved!"<<endl;
      constk=x.constk;
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
      observable(this)
    {}

    AtomsPackObj(const cnine::Tensor<int>& M):
      AtomsPackObj(cnine::array_pool<int>(M)){
      PTENS_ASSRT(M.ndims()==2);
      constk=M.dim(1);
    }


  public: // ---- Views --------------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


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


  public: // ---- to_nodes_map -------------------------------------------------------------------------------


    cnine::oncep<cnine::GatherMap> gather_to_nodes_map=
      cnine::oncep<cnine::GatherMap>([&](){

	cnine::map_of_lists2<int,int> lists;
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

    
    // create map for messages from y
    TransferMap overlaps(const AtomsPackObj& y){
      return overlap_maps(y);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPackObj";
    }

    friend ostream& operator<<(ostream& stream, const AtomsPackObj& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
    //overlap_maps([this](const AtomsPackObj& x)
    //{return TransferMap(new TransferMapObj<AtomsPackObj>(x,*this));}),
    //cat_maps([this](const vector<AtomsPackObj*>& v)
    //{return shared_ptr<AtomsPackObj>(new AtomsPackObj(cat_with(v)));})
		//overlap_maps([this](const AtomsPackObj& x)
		//{return TransferMap(new TransferMapObj<AtomsPackObj>(x,*this));}),
		//cat_maps([this](const vector<AtomsPackObj*>& v)
		//{return shared_ptr<AtomsPackObj>(new AtomsPackObj(cat_with(v)));})
		//overlap_maps([this](const AtomsPackObj& x)
		//{return TransferMap(new TransferMapObj<AtomsPackObj>(x,*this));}),
		//cat_maps([this](const vector<AtomsPackObj*>& v)
		//{return shared_ptr<AtomsPackObj>(new AtomsPackObj(cat_with(v)));})
    //cnine::once<BASE> gpack=cnine::once<BASE>([&](){
    //BASE R(*this,1);
    //R.dir.move_to_device(1);
    //return R;
    //});


    //cnine::once<BASE> gpack=cnine::once<BASE>([&](){
    //BASE R(*this,1);
    //R.dir.move_to_device(1);
    //return R;
    //});


