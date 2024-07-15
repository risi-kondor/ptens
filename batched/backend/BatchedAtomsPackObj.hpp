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


#ifndef _ptens_BatchedAtomsPackObj
#define _ptens_BatchedAtomsPackObj

#include "shared_object_pack.hpp"
#include "AtomsPackObj.hpp"


namespace ptens{


  class BatchedAtomsPackObj: public cnine::shared_object_pack<AtomsPackObj>{
  public:

    typedef cnine::shared_object_pack<AtomsPackObj> BASE;

    using BASE::BASE;
    //using BASE::size;
    //using BASE::operator[];

    mutable vector<int> row_offsets0;
    mutable vector<int> row_offsets1;
    mutable vector<int> row_offsets2;


  public: // ---- Constructors -------------------------------------------------------------------------------


    BatchedAtomsPackObj(const vector<vector<vector<int> > >& v){
      for(auto& p:v)
	push_back(to_share(new AtomsPackObj(p)));
    }

    BatchedAtomsPackObj(const initializer_list<initializer_list<initializer_list<int> > >& v){
      for(auto& p:v)
	push_back(to_share(new AtomsPackObj(p)));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    vector<vector<vector<int > > > as_vecs() const{
      vector<vector<vector<int > > > R;
      for(int i=0; i<size(); i++)
	R.push_back((*this)[i].as_vecs());
      return R;
    }


  public: // ---- 0th order layout -----------------------------------------------------------------------------------


    int nrows0() const{
      if(row_offsets0.size()==0) make_row_offsets0();
      return row_offsets0.back();
    }

    int offset0(const int i) const{
      if(i==0) return 0;
      if(row_offsets0.size()==0) make_row_offsets0();
      return row_offsets0[i-1];
    }

    int nrows0(const int i) const{
      if(row_offsets0.size()==0) make_row_offsets0();
      if(i==0) return row_offsets0[0];
      return row_offsets0[i]-row_offsets0[i-1];
    }

    void make_row_offsets0() const{
      row_offsets0.resize(size());
      int t=0;
      for(int i=0; i<size(); i++){
	t+=(*this)[i].nrows0();
	  row_offsets0[i]=t;
	}
    }


  public: // ---- 1st order layout -----------------------------------------------------------------------------------


    int nrows1() const{
      if(row_offsets1.size()==0) make_row_offsets1();
      return row_offsets1.back();
    }

    int offset1(const int i) const{
      if(i==0) return 0;
      if(row_offsets1.size()==0) make_row_offsets1();
      return row_offsets1[i-1];
    }

    int nrows1(const int i) const{
      if(row_offsets1.size()==0) make_row_offsets1();
      if(i==0) return row_offsets1[0];
      return row_offsets1[i]-row_offsets1[i-1];
    }

    void make_row_offsets1() const{
      row_offsets1.resize(size());
      int t=0;
      for(int i=0; i<size(); i++){
	t+=(*this)[i].nrows1();
	  row_offsets1[i]=t;
	}
    }


  public: // ---- 2nd order layout -----------------------------------------------------------------------------------


    int nrows2() const{
      if(row_offsets2.size()==0) make_row_offsets2();
      return row_offsets2.back();
    }

    int offset2(const int i) const{
      if(i==0) return 0;
      if(row_offsets2.size()==0) make_row_offsets2();
      return row_offsets2[i-1];
    }

    int nrows2(const int i) const{
      if(row_offsets0.size()==0) make_row_offsets2();
      if(i==0) return row_offsets2[0];
      return row_offsets2[i]-row_offsets2[i-1];
    }

    void make_row_offsets2() const{
      row_offsets2.resize(size());
      int t=0;
      for(int i=0; i<size(); i++){
	t+=(*this)[i].nrows2();
	  row_offsets2[i]=t;
	}
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedAtomsPackObj";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"[";
      for(int i=0; i<size(); i++){
	oss<<(*this)[i];
	if(i<size()-1) oss<<",";
      }
      oss<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BatchedAtomsPackObj& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
