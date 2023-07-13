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
 *
 */
#ifndef _ptens_Ptensor1pack
#define _ptens_Ptensor1pack

#include "Cgraph.hpp"
//#include "Ptensor1subpack.hpp"
//#include "PtensorSubpackSpecializer.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor1.hpp"


namespace ptens{


  class Ptensor1pack: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc;
    AtomsPack atoms;

    ~Ptensor1pack(){
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensor1pack(){}

    Ptensor1pack(const int _nc, const int _dev=0):
      RtensorPool(_dev), nc(_nc){}


  public: // ----- Constructors ------------------------------------------------------------------------------


    static Ptensor1pack raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor1pack R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor1::raw(_atoms(i),_nc));
      }
      return R;
    }

    static Ptensor1pack zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor1pack R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor1::zero(_atoms(i),_nc));
      }
      return R;
    }


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensor1pack(const Ptensor1pack& x):
      RtensorPool(x),
      atoms(x.atoms){}
	
    Ptensor1pack(Ptensor1pack&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)){}

    Ptensor1pack& operator=(const Ptensor1pack& x)=delete;


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }

    int push_back(const Ptensor1& x){
      if(size()==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    Ptensor1pack fwd(const Cgraph& graph) const{
      Ptensor1pack R;
      for(int i=0; i<graph.n; i++) //TODO
	R.push_back(Ptensor1::zero(atoms_of(i),5*2));
      R.forwardMP(*this,graph);
      return R;
    }


    void forwardMP(const Ptensor1pack& x, const Cgraph& graph){
      AindexPack src_indices;
      AindexPack dest_indices;

      graph.forall_edges([&](const int i, const int j){
	  Atoms atoms0=atoms_of(i);
	  Atoms atoms1=atoms_of(j);
	  Atoms intersect=atoms0.intersect(atoms1);
	  src_indices.push_back(i,atoms0(intersect));
	  dest_indices.push_back(j,atoms1(intersect));
	});

      RtensorPool messages0=x.messages0(src_indices);
      add_messages0(messages0,dest_indices,0);
      //cout<<messages0<<endl;

      RtensorPool messages1=x.messages1(src_indices);
      add_messages1(messages1,dest_indices,5); // TODO 
      //cout<<messages1<<endl;

    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool messages0(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++){
	Rtensor2_view source=view2_of(src_list.tix(i));
	Rtensor1_view dest=R.view1_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=dest.n0;

	for(int c=0; c<nc; c++){
	  float t=0; 
	  for(int j=0; j<n; j++) 
	    t+=source(ix[j],c);
	  dest.set(c,t);
	}
      }

      return R;
    }


    RtensorPool messages1(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({src_list.nindices(i),dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++){
	Rtensor2_view source=view2_of(src_list.tix(i));
	Rtensor2_view dest=R.view2_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=dest.n1;

	for(int c=0; c<nc; c++){
	  for(int j=0; j<n; j++) 
	    dest.set(j,c,source(ix[j],c));
	}
      }

      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void add_messages0(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++){
	Rtensor1_view source=messages.view1_of(i);
	Rtensor2_view dest=view2_of(dest_list.tix(i));
	vector<int> ix=dest_list.indices(i);
	int n=ix.size();
	int nc=source.n0;

	for(int c=0; c<nc; c++){
	  float v=source(c);
	  for(int j=0; j<n; j++) 
	    dest.inc(ix[j],c+coffs,v);
	}
      }
    }


    void add_messages1(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++){
	Rtensor2_view source=messages.view2_of(i);
	Rtensor2_view dest=view2_of(dest_list.tix(i));
	vector<int> ix=dest_list.indices(i);
	int n=ix.size();
	int nc=source.n1;

	for(int c=0; c<nc; c++){
	  for(int j=0; j<n; j++) 
	    dest.inc(ix[j],c+coffs,source(j,c));
	}
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Ptensor "<<i<<" "<<Atoms(atoms(i))<<":"<<endl;
	oss<<RtensorPool::operator()(i).str()<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor1pack& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
