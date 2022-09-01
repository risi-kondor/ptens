#ifndef _ptens_Ptensor
#define _ptens_Ptensor

#include "Atoms.hpp"
#include "RtensorObj.hpp"

#define PTENSOR_PTENSOR_IMPL cnine::RtensorObj


namespace ptens{

  class Ptensor1: public PTENSOR_PTENSOR_IMPL{
  public:

    Atoms atoms;


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


    float at_(const int i, const int c) const{
      return value(atoms(i),c);
    }

    void inc_(const int i, const int c, float x){
      inc(atoms(i),c,x);
    }


    // ---- Message passing ----------------------------------------------------------------------------------


    Ptensor1(const Ptensor1& x, const Atoms& _atoms):
      Ptensor1(_atoms,5,cnine::fill_zero()){

      Atoms cap=atoms.intersect(x.atoms);
      int k=cap.size();

      float s=0;
      for(int i=0; i<dims[0]; i++)
	s+=x.value(i,0);

      float t=0;
      for(int i=0; i<k; i++){
	int a=cap[i];
	inc_(a,0,x.at_(a,0));
	t+=x.at_(a,0);
      }
      
      for(int i=0; i<k; i++){
	inc_(cap[i],1,t);
	inc_(cap[i],2,s);
      }

      for(int i=0; i<dims[0]; i++){
	inc(i,3,t);
	inc(i,4,s);
      }

    }


    void gather(const Ptensor1& x){
      
      //const int 

    }


  };


}


#endif 
