#ifndef _ptens_Ptensor0
#define _ptens_Ptensor0

#include "Atoms.hpp"
#include "RtensorA.hpp"
#include "RtensorObj.hpp"
//#include "PtensorSgntr.hpp"


namespace ptens{

  class Ptensor0: public cnine::RtensorA{
  public:


    typedef cnine::RtensorA rtensor;

    Atoms atoms;

    #ifdef WITH_FAKE_GRAD
    Ptensor0* grad=nullptr;
    #endif 


    ~Ptensor0(){
      #ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
      #endif 
    }


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor0(const Atoms& _atoms, const int nc, const FILLTYPE& dummy, const int _dev=0):
      rtensor(cnine::Gdims(nc),dummy,_dev),
      atoms(_atoms){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor0 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor0 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor0 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor0 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor0(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    Ptensor0(RtensorA&& x, Atoms&& _atoms):
      RtensorA(std::move(x)),
      atoms(std::move(_atoms)){}
 

    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return dims(0);
    }

    int get_nc() const{
      return dims.back();
    }

    //PtensorSgntr signature() const{
    //return PtensorSgntr(getk(),get_nc());
    //}

    float at_(const int i, const int c) const{
      return (*this)(atoms(i),c);
    }

    void inc_(const int i, const int c, float x){
      inc(atoms(i),c,x);
    }


    // ---- Message passing ----------------------------------------------------------------------------------


    Ptensor0(const Ptensor0& x, const Atoms& _atoms):
      Ptensor0(_atoms,5*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }


    void pull_msg(const Ptensor0& x){
      int nc=x.get_nc();
      assert(get_nc()==5*nc);

      Atoms common=atoms.intersect(x.atoms);
      int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int j=0; j<nc; j++){

	float s=0;
	for(int i=0; i<dims[0]; i++)
	  s+=x(i,j);
     
	float t=0;
	for(int i=0; i<k; i++){
	  t+=x(xix[i],j);
	}

	for(int i=0; i<k; i++){
	  inc(ix[i],5*j,x(xix[i],j));
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

    /*
    rtensor reductions0(const vector<int>& ix, const int c) const{
      const int n=dim(0);
      const int k=ix.size();
      rtensor R=rtensor::raw(cnine::Gdims(get_nc()));
      for(int i=0; i<
      {float t=0; for(int i=0; i<k; i++) t+=(*this)(ix[i],c); R.set(0,t);}
      {float t=0; for(int i=0; i<n; i++) t+=(*this)(i,c); R.set(1,t);}
      return R;
    }
    */

  public: // ---- Broadcasting -------------------------------------------------------------------------------

    /*
   void broadcast0(const rtensor& R0, const vector<int>& ix, int coffs){
     assert(R0.ndims()==1);

      for(int c=0; c<R0.dim(0); c++){
	inc(coffs,R0(s));
	coffs+=2;
      }
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="")const{
      ostringstream oss;
      oss<<indent<<"Ptensor0"<<atoms<<":"<<endl;
      oss<<rtensor::str(indent);
      return oss.str();
    }


  };

}


#endif 

