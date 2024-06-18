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
#ifndef _ptens_Ptensor2
#define _ptens_Ptensor2

#include "Atoms.hpp"
#include "Ptensor2.hpp"



namespace ptens{

  class Ptensor3: public PTENSOR_PTENSOR_IMPL{
  public:

    Atoms atoms;

    typedef cnine::Ltensor<float> BASE;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor3(const Atoms& _atoms, const int nc, const FILLTYPE& dummy, const int _dev=0):
      PTENSOR_PTENSOR_IMPL(cnine::Gdims(_atoms.size(),_atoms.size(),_atoms.size(),nc),dummy,_dev),
      atoms(_atoms){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor3 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor3(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor3 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor3(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor3 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor3(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor3 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor3(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Access -------------------------------------------------------------------------------------------


    int get_nc() const{
      return dims.back();
    }

    float at_(const int i0, const int i1, const int i2, const int c) const{
      return value(atoms(i0),atoms(i1),atoms(i2),c);
    }

    void inc_(const int i0, const int i1, const int i2, const int c, float x){
      inc(atoms(i0),atoms(i1),atoms(i2),c,x);
    }


    // ---- Message passing ----------------------------------------------------------------------------------


    Ptensor3(const Ptensor3& x, const Atoms& _atoms):
      Ptensor3(_atoms,52*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor3(const Ptensor2& x, const Atoms& _atoms):
      Ptensor3(_atoms,52*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor3(const Ptensor1& x, const Atoms& _atoms):
      Ptensor3(_atoms,15*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor1 msg1(const Atoms& _atoms){
      Ptensor1 R(_atoms,15*get_nc(),cnine::fill_zero());
      push_msg(R);
      return R;
    }

    Ptensor2 msg2(const Atoms& _atoms){
      Ptensor1 R(_atoms,15*get_nc(),cnine::fill_zero());
      push_msg(R);
      return R;
    }


    void pull_msg(const Ptensor1& x){
      int nc=x.get_nc();
      const int cstride=15;
      assert(get_nc()==cstride*nc);

      Atoms common=atoms.intersect(x.atoms);
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int c=0; c<nc; c++){
	rtensor R1=x.reductions1(xix,c);
	broadcast1(R1,ix,cstride*c);
	rtensor R0=x.reductions0(xix,c);
	broadcast0(R0,ix,cstride*c+5);
      }
      
    }


    void pull_msg(const Ptensor2& x){
      int nc=x.get_nc();
      const int cstride=52;
      assert(get_nc()==cstride*nc);

      Atoms common=atoms.intersect(x.atoms);
      int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));


      for(int c=0; c<nc; c++){

	for(int i=0; i<k; i++)
	  for(int j=0; j<k; j++){
	    inc(ix[i],ix[j],cstride*c,x.value(xix[i],xix[j],c));
	    inc(ix[j],ix[i],cstride*c+1,x.value(xix[i],xix[j],c));
	}

	rtensor R1=x.reductions1(xix,c);
	broadcast1(R1,ix,cstride*c+2);

	rtensor R0=x.reductions0(R1,xix,c);
	broadcast0(R0,ix,cstride*c+27);

      }
      
    }


    void push_msg(Ptensor1& x) const{
      int nc=get_nc();
      const int cstride=15;
      assert(x.get_nc()==cstride*nc);

      Atoms common=atoms.intersect(x.atoms);
      //int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int c=0; c<nc; c++){
	rtensor R1=reductions1(ix,c);
	x.broadcast1(R1,xix,cstride*c);
	rtensor R0=reductions0(R1,ix,c);
	x.broadcast0(R0,xix,cstride*c+5);
      }
      
    }


    // ---- Reductions ---------------------------------------------------------------------------------------


    rtensor reductions3(const vector<int>& ix, const int c) const{
      const int k=ix.size();
      rtensor R=rtensor::raw({k,k,k,1});
      
      for(int i=0; i<k; i++){
	int _i=ix[i];
	for(int j=0; j<k; j++){
	  int _j=ix[j];
	  for(int l=0; l<k; l++){
	    int _l=ix[l];
	    R.set(i,j,l,0,value(_i,_j,_l,c));
	  }
	}
      }

      return R;
    }

    rtensor reductions1(const vector<int>& ix, const int c) const{
      const int k=ix.size();
      const int n=dims(0);
      rtensor R=rtensor::raw({k,5});
      
      for(int i=0; i<k; i++){
	int _i=ix[i];
	{float s=0; for(int j=0; j<k; j++) s+=value(_i,ix[j],c); R.set(i,0,s);}
	{float s=0; for(int j=0; j<k; j++) s+=value(ix[j],_i,c); R.set(i,1,s);}
	{float s=0; for(int j=0; j<n; j++) s+=value(_i,j,c); R.set(i,2,s);}
	{float s=0; for(int j=0; j<n; j++) s+=value(j,_i,c); R.set(i,3,s);}
	R.set(i,4,value(_i,_i,c));
      }

      return R;
    }


    rtensor reductions0(const rtensor& R1, const vector<int>& ix, const int c) const{
      const int k=ix.size();
      const int n=dims(0);
      rtensor R=rtensor::zero({5});

      for(int i=0; i<k; i++){
	  R.inc(0,R1(i,0));
	  R.inc(2,R1(i,2));
	  R.inc(3,R1(i,3));
	  R.inc(4,R1(i,4));
	}

      float t=0;
      for(int i=0; i<n; i++)
	for(int j=0; j<n; j++)
	  t+=value(i,j,c);
      R.inc(1,t);
      
      return R;
    }

    
  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast3(const rtensor& R3, const vector<int>& ix, int coffs){
      const int k=ix.size();
      //const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++){
	    int _j=ix[j];
	    for(int l=0; l<k; l++){
	      int _l=ix[l];
	      float v=R2.value(i,j,l,s);
	      inc(_i,_j,_l,coffs,v);
	      inc(_i,_l,_j,coffs+1,v);
	      inc(_j,_i,_l,coffs+2,v);
	      inc(_j,_l,_i,coffs+3,v);
	      inc(_l,_i,_j,coffs+4,v);
	      inc(_l,_j,_i,coffs+5,v);
	    }
	  }
	}
	coffs+=6;
      }
    }

    void broadcast2(const rtensor& R2, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++){
	    int _j=ix[j];
	    float v=R2.value(i,j,s);

	    for(int l=0; l<k; l++){
	      int _l=ix[l];
	      inc(_i,_j,_l,coffs,v);
	      inc(_j,_i,_l,coffs+1,v);
	      inc(_i,_l,_j,coffs+2,v);
	      inc(_j,_l,_i,coffs+3,v);
	      inc(_l,_i,_j,coffs+4,v);
	      inc(_l,_j,_i,coffs+5,v);
	    }

	    for(int l=0; l<n; l++){
	      inc(_i,_j,l,coffs+6,v);
	      inc(_j,_i,l,coffs+7,v);
	      inc(_i,l,_j,coffs+8,v);
	      inc(_j,l,_i,coffs+9,v);
	      inc(l,_i,_j,coffs+10,v);
	      inc(l,_j,_i,coffs+11,v);
	    }

	    inc(_i,_i,_j,coffs+12,v);
	    inc(_j,_j,_i,coffs+13,v);
	    inc(_i,_j,_i,coffs+14,v);
	    inc(_j,_i,_j,coffs+15,v);
	    inc(_j,_i,_i,coffs+16,v);
	    inc(_i,_j,_j,coffs+17,v);

	  }
	}
	coffs+=18;
      }
    }


    void broadcast1(const rtensor& R1, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  float v=R1.value(i,s);

	  for(int j=0; j<k; j++){
	    int _j=ix[j];
	    for(int l=0; l<k; l++){
	      int _l=ix[l];
	      inc(_i,_j,_l,coffs,v);
	      inc(_j,_i,_l,coffs+1,v);
	      inc(_j,_l,_i,coffs+2,v);
	    }
	  }

	  for(int j=0; j<n; j++){
	    for(int l=0; l<k; l++){
	      int _l=ix[l];
	      inc(_i,j,_l,coffs+3,v);
	      inc(j,_i,_l,coffs+4,v);
	      inc(j,_l,_i,coffs+5,v);
	      inc(_i,_l,j,coffs+6,v);
	      inc(l,_i,_j,coffs+7,v);
	      inc(l,_j,_i,coffs+8,v);
	    }
	  }

	  for(int j=0; j<n; j++){
	    for(int l=0; l<n; l++){
	      int _l=ix[l];
	      inc(_i,j,l,coffs+9,v);
	      inc(j,_i,l,coffs+10,v);
	      inc(j,l,_i,coffs+11,v);
	    }
	  }

	  for(int j=0; j<k; j++){
	    inc(ix[j],ix[j],_i,coffs+12,v);
	    inc(ix[j],_i,ix[j],coffs+13,v);
	    inc(_i,ix[j],ix[j],coffs+14,v);
	  }

	  for(int j=0; j<n; j++){
	    inc(j,j,_i,coffs+15,v);
	    inc(j,_i,j,coffs+16,v);
	    inc(_i,j,j,coffs+17,v);
	  }

	  inc(_i,_i,_i,coffs+18,R1.value(i,s));
	}
	coffs+=19;
      }
    }
    

    void broadcast0(const rtensor& R0, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R0.dim(0); s++){
	float v=R0.value(s);

	for(int i=0; i<k; i++)
	  for(int j=0; j<k; j++)
	    for(int l=0; l<k; l++)
	      inc(ix[i],ix[j],ix[l],coffs,v);

	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++){
	    int _j=ix[j];
	    for(int l=0; l<n; l++){
	      inc(_i,_j,l,coffs+1,v);
	      inc(_i,l,_j,coffs+2,v);
	      inc(l,_i,_j,coffs+3,v);
	    }
	  }
	}

	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<n; j++){
	    for(int l=0; l<n; l++){
	      inc(_i,j,l,coffs+4,v);
	      inc(j,_i,l,coffs+5,v);
	      inc(j,l,_i,coffs+6,v);
	    }
	  }
	}

	for(int i=0; i<n; i++)
	  for(int j=0; j<n; j++)
	    for(int l=0; l<n; l++)
	      inc(i,j,l,coffs+7,v);

	coffs+=8;
      }      
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------



  };


}


#endif 
