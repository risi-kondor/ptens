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

#include "AtomsPackObj.hpp"


namespace ptens{

  class AtomsPack{
  public:

    shared_ptr<AtomsPackObj> obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPack():
      obj(new AtomsPackObj()){}

    AtomsPack(AtomsPackObj* _obj):
      obj(_obj){}

    AtomsPack(shared_ptr<AtomsPackObj> _obj):
      obj(_obj){}

    AtomsPack(const int N):
      obj(new AtomsPackObj(N)){}

    AtomsPack(const int N, const int k):
      obj(new AtomsPackObj(N,k)){}

    AtomsPack(const vector<vector<int> >& x):
      obj(new AtomsPackObj(x)){}

    AtomsPack(const initializer_list<initializer_list<int> >& x):
      obj(new AtomsPackObj(x)){}

    AtomsPack(const cnine::labeled_forest<int>& forest):
      obj(new AtomsPackObj(forest)){}


  public: // ---- Static Constructors ------------------------------------------------------------------------


    static AtomsPack random(const int n, const float p=0.5){
      return AtomsPack(new AtomsPackObj(AtomsPackObj::random(n,p)));
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    //AtomsPack& operator=(const AtomsPack& x){
      //cout<<"AtomsPAck::operator="<<endl;
      //obj=x.obj;
      //return *this;
    //}

  public: // ---- Conversions --------------------------------------------------------------------------------


    AtomsPack(cnine::array_pool<int>&& x):
      obj(new AtomsPackObj(std::move(x))){}

    AtomsPack(const cnine::Tensor<int>& M):
      obj(new AtomsPackObj(M)){}


  public: // ---- Views --------------------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    int constk() const{
      return obj->constk;
    }

    int size_of(const int i) const{
      return obj->size_of(i);
    }

    vector<int> operator()(const int i) const{
      return (*obj)(i);
    }

    Atoms operator[](const int i) const{
      return (*obj)[i];
    }

    void push_back(const vector<int>& v){
      obj->push_back(v);
    }

    void push_back(const set<int>& v){
      obj->push_back(v);
    }

    void push_back(const initializer_list<int>& v){
      push_back(vector<int>(v));
    }

    int tsize0() const{
      return obj->tsize0();
    }

    int tsize1() const{
      return obj->tsize1();
    }

    int tsize2() const{
      return obj->tsize2();
    }

    vector<vector<int> > as_vecs() const{
      return obj->as_vecs();
    }

    bool operator==(const AtomsPack& x) const{
      return (*obj)==(*x.obj);
    }

    cnine::array_pool<int> dims1(const int nc) const{
      return obj->dims1(nc);
    }

    cnine::array_pool<int> dims2(const int nc) const{
      return obj->dims1(nc);
    }

    pair<int*,int*> gpu_arrs(const int dev){
      return obj->gpu_arrs(dev);
    }

    int* gather_to_nodes_map(const int dev){
      return obj->gather_to_nodes_map()->arrg;
    }


  public: // ---- Concatenation ------------------------------------------------------------------------------

    
    static AtomsPack cat(const vector<reference_wrapper<AtomsPack> >& list){
      return AtomsPackObj::cat
	(cnine::mapcar<reference_wrapper<AtomsPack>,reference_wrapper<AtomsPackObj> >
	  (list,[](const reference_wrapper<AtomsPack>& x){
	    return reference_wrapper<AtomsPackObj>(*x.get().obj);}));
    }
    

  public: // ---- Operations ---------------------------------------------------------------------------------


    AtomsPack permute(const cnine::permutation& pi){
      return AtomsPack(new AtomsPackObj(obj->permute(pi)));
    } 
    
    // create map for messages from y
    TransferMap overlaps(const AtomsPack& y){
      return obj->overlaps(*y.obj);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "AtomsPack";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const AtomsPack& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
