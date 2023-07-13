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
#ifndef _ptens_Ptensor0pack
#define _ptens_Ptensor0pack

#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor0.hpp"


namespace ptens{


  class Ptensor0pack: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc=0;
    AtomsPack atoms;

    ~Ptensor0pack(){
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensor0pack(){}

    Ptensor0pack(const int _nc, const int _dev=0):
      RtensorPool(_dev), nc(_nc){}

    Ptensor0pack(const int _nc, const int _memsize, const int _dev):
      RtensorPool(_dev), nc(_nc){
      reserve(_memsize);
    }
    
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor0pack(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_n, cnine::Gdims({_nc}), dummy, _dev), atoms(_n), nc(_nc){}

    Ptensor0pack(const int _n, const int _nc, const cnine::fill_sequential& dummy, const int _dev=0):
      RtensorPool(_n,cnine::Gdims({_nc}),cnine::fill_raw(),_dev), atoms(_n), nc(_nc){
      for(int i=0; i<_n; i++)
	view1_of(i).set(i);
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    static Ptensor0pack raw(const int _n, const int _nc, const int _dev=0){
      return Ptensor0pack(_n,_nc,cnine::fill_raw(),_dev);}
    static Ptensor0pack zero(const int _n, const int _nc, const int _dev=0){
      return Ptensor0pack(_n,_nc,cnine::fill_zero(),_dev);}
    static Ptensor0pack sequential(const int _n, const int _nc, const int _dev=0){
      return Ptensor0pack(_n,_nc,cnine::fill_sequential(),_dev);}

    /*
    static Ptensor0pack raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor0pack R(_nc,_dev);
      R.reserve(_atoms.tsize0()*_nc);

      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor0::raw(_atoms(i),_nc));
      }
      return R;
    }
    static Ptensor0pack zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor0pack R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor0::zero(_atoms(i),_nc));
      }
      return R;
    }
    static Ptensor0pack sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor0pack R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor0::zero(_atoms(i),_nc));
	auto A=R.view1_of(i);
	for(int j=0; j<_nc; j++) A.set(j,i);
      }
      return R;
    }
    */


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensor0pack(const Ptensor0pack& x):
      RtensorPool(x),
      atoms(x.atoms){}
	
    Ptensor0pack(Ptensor0pack&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)){}

    Ptensor0pack& operator=(const Ptensor0pack& x)=delete;


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }
    
    
    void push_back(const Ptensor0& x){
      if(nc==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
    }

    //void push_back_zero(const Gdims& 



  public: // ---- Message passing ----------------------------------------------------------------------------



  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool messages0(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++)
	R.view1_of(i)=view1_of(src_list.tix(i));

      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void add_messages0(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++)
	view1_of(dest_list.tix(i))=messages.view1_of(i);
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

    friend ostream& operator<<(ostream& stream, const Ptensor0pack& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
