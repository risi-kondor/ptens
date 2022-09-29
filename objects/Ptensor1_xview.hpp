#ifndef _Ptensor1_xview
#define _Ptensor1_xview

#include "Rtensor2_view.hpp"

namespace ptens{

  class Ptensor1_xview: public cnine::Rtensor2_view{
  public:

    typedef cnine::RtensorA rtensor;
    typedef cnine::Gdims Gdims;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;


  public:

    const vector<int> ix;


  public:

    Ptensor1_xview(float* _arr, const int _nc, const int _s0, const int _s1, 
      const vector<int>& _ix, const int _dev=0): 
      Rtensor2_view(_arr,_ix.size(),_nc,_s0,_s1,_dev),
      ix(_ix){}
    //arr(_arr), n0(_n0), n1(_n1), s0(_s0), s1(_s1), ix(_ix), dev(_dev){}


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_regular() const{
      return false;
    }

    float operator()(const int i0, const int i1) const{
      return arr[s0*ix[i0]+s1*i1];
    }

    float get(const int i0, const int i1) const{
      return arr[s0*ix[i0]+s1*i1];
    }

    
    void set(const int i0, const int i1, float x) const{
      arr[s0*ix[i0]+s1*i1]=x;
    }

    void inc(const int i0, const int i1, float x) const{
      arr[s0*ix[i0]+s1*i1]+=x;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const Rtensor2_view& x) const{
      CNINE_CPUONLY();
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  arr[s0*ix[i0]+s1*i1]+=x.arr[x.s0*i0+x.s1*i1];
    }

    void operator+=(const Rtensor2_view& x) const{
      return add(x);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    void sum0_into(const Rtensor1_view& r){
      CNINE_CPUONLY();
      assert(r.n0==n1);
      for(int i=0; i<n1; i++){
	float t=0; 
	for(int j=0; j<n0; j++) 
	  t+=arr[s0*ix[j]+s1*i];
	r.inc(i,t);
      }
    }


  };



}

#endif
