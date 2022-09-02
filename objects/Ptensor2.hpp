#ifndef _ptens_Ptensor2
#define _ptens_Ptensor2

#include "Atoms.hpp"
#include "RtensorObj.hpp"
#include "Ptensor1.hpp"

// #define PTENSOR_PTENSOR_IMPL cnine::RtensorObj


namespace ptens{

  class Ptensor2: public PTENSOR_PTENSOR_IMPL{
  public:

    Atoms atoms;

    typedef cnine::RtensorObj rtensor;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor2(const Atoms& _atoms, const int nc, const FILLTYPE& dummy, const int _dev=0):
      PTENSOR_PTENSOR_IMPL(cnine::Gdims(_atoms.size(),_atoms.size(),nc),dummy,_dev),
      atoms(_atoms){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor2 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor2 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor2 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor2 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Access -------------------------------------------------------------------------------------------


    int get_nc() const{
      return dims.back();
    }

    float at_(const int i, const int j, const int c) const{
      return value(atoms(i),atoms(j),c);
    }

    void inc_(const int i, const int j, const int c, float x){
      inc(atoms(i),atoms(j),c,x);
    }


    // ---- Message passing ----------------------------------------------------------------------------------


    Ptensor2(const Ptensor2& x, const Atoms& _atoms):
      Ptensor2(_atoms,52*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor2(const Ptensor1& x, const Atoms& _atoms):
      Ptensor2(_atoms,15*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor1 msg1(const Atoms& _atoms){
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


    void broadcast1(const rtensor& R1, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++)
	    inc(_i,ix[j],coffs,R1.value(i,s));
	  for(int j=0; j<k; j++)
	    inc(ix[j],_i,coffs+1,R1.value(i,s));
	  for(int j=0; j<n; j++)
	    inc(_i,j,coffs+2,R1.value(i,s));
	  for(int j=0; j<n; j++)
	    inc(j,_i,coffs+3,R1.value(i,s));
	  inc(_i,_i,coffs+4,R1.value(i,s));
	}
	coffs+=5;
      }
    }
    

    void broadcast0(const rtensor& R0, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R0.dim(0); s++){
	float v=R0.value(s);

	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++)
	    inc(_i,ix[j],coffs,v);
	  for(int j=0; j<n; j++){
	    inc(_i,j,coffs+1,v);
	    inc(j,_i,coffs+2,v);
	  }
	  inc(_i,_i,coffs+3,v);
	}

	for(int i=0; i<n; i++)
	  for(int j=0; j<n; j++)
	    inc(i,j,coffs+4,v);
	coffs+=5;

      }      
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------



  };


}


#endif 
