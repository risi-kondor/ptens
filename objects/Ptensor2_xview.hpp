#ifndef _PTensor2_xview
#define _PTensor2_xview

#include "Rtensor2_view.hpp"

namespace ptens{

  class PTensor2_xview: public cnine::Rtensor2_view{
  public:

    typedef cnine::RtensorA rtensor;
    typedef cnine::Gdims Gdims;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;


  public:

    const vector<int> ix;


  public:

    PTensor2_xview(float* _arr, const int _nc, const int _s0, const int _s1, const int _s2, 
      const vector<int>& _ix, const int _dev=0): 
      Rtensor2_view(_arr,_ix.size(),_ix.size(),_nc,_s0,_s1,s2,_dev),
      ix(_ix){}


  public: // ---- Access -------------------------------------------------------------------------------------


    float operator()(const int i0, const int i1) const{
      return arr[s0*ix[i0]+s1*i1];
    }

    /*
    void set(const int i0, const int i1, float x) const{
    }

    void inc(const int i0, const int i1, float x) const{
    }
    */


  public: // ---- Operations --------------------------------------------------------------------------------


  };

}

#endif
