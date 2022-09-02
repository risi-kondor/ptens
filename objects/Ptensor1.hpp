#ifndef _ptens_Ptensor1
#define _ptens_Ptensor1

#include "Atoms.hpp"
#include "RtensorObj.hpp"

#define PTENSOR_PTENSOR_IMPL cnine::RtensorObj


namespace ptens{

  class Ptensor1: public PTENSOR_PTENSOR_IMPL{
  public:

    Atoms atoms;

    typedef cnine::RtensorObj rtensor;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor1(const Atoms& _atoms, const int nc, const FILLTYPE& dummy, const int _dev=0):
      PTENSOR_PTENSOR_IMPL(cnine::Gdims(_atoms.size(),nc),dummy,_dev),
      atoms(_atoms){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor1 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor1 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor1 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor1 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Access -------------------------------------------------------------------------------------------


    int get_nc() const{
      return dims.back();
    }

    float at_(const int i, const int c) const{
      return value(atoms(i),c);
    }

    void inc_(const int i, const int c, float x){
      inc(atoms(i),c,x);
    }


    // ---- Message passing ----------------------------------------------------------------------------------


    Ptensor1(const Ptensor1& x, const Atoms& _atoms):
      Ptensor1(_atoms,5*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }


    void pull_msg(const Ptensor1& x){
      int nc=x.get_nc();
      assert(get_nc()==5*nc);

      Atoms common=atoms.intersect(x.atoms);
      int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int j=0; j<nc; j++){

	float s=0;
	for(int i=0; i<dims[0]; i++)
	  s+=x.value(i,j);
     
	float t=0;
	for(int i=0; i<k; i++){
	  t+=x.value(xix[i],j);
	}

	for(int i=0; i<k; i++){
	  inc(ix[i],5*j,x.value(xix[i],j));
	  inc(ix[i],5*j+1,t);
	  inc(ix[i],5*j+2,s);
	}

	for(int i=0; i<dims[0]; i++){
	  inc(i,5*j+3,t);
	  inc(i,5*j+4,s);
	}

      }

    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    rtensor reductions1(const vector<int>& ix, const int c) const{
      const int k=ix.size();
      rtensor R=rtensor::raw(cnine::Gdims(k));
      for(int i=0; i<k; i++)
	R.set(i,value(ix[i],c));
      return R;
    }

    rtensor reductions0(const vector<int>& ix, const int c) const{
      const int n=dim(0);
      const int k=ix.size();
      rtensor R=rtensor::raw(cnine::Gdims(2));
      {float t=0; for(int i=0; i<k; i++) t+=value(ix[i],c); R.set(0,t);}
      {float t=0; for(int i=0; i<n; i++) t+=value(i,c); R.set(1,t);}
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast1(const rtensor& R1, const vector<int>& ix, int coffs){
      const int k=ix.size();
      //const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++)
	  inc(ix[i],coffs,R1.value(i,s));
	coffs+=1;
      }
    }

    
   void broadcast0(const rtensor& R0, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R0.dim(1); s++){
	for(int i=0; i<k; i++)
	  inc(ix[i],coffs,R0.value(i,s));
	for(int i=0; i<n; i++)
	  inc(i,coffs+1,R0.value(i,s));
	coffs+=2;
      }
    }
    




  };


}


#endif 
