#ifndef _ptens_LinmapFunctions
#define _ptens_LinmapFunctions

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"


namespace ptens{
  
  inline Ptensor0 linmaps0(const Ptensor0& x){
    Ptensor0 R=Ptensor0::zero(x.atoms,x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensor0 linmaps0(const Ptensor1& x){
    Ptensor0 R=Ptensor0::zero(x.atoms,x.nc,x.dev);
    x.add_linmaps_to(R);
    return R;
  }

  inline Ptensor0 linmaps0(const Ptensor2& x){
    Ptensor0 R=Ptensor0::zero(x.atoms,2*x.nc,x.dev);
    x.add_linmaps_to(R);
    return R;
  }


  inline Ptensor1 linmaps1(const Ptensor0& x){
    Ptensor1 R=Ptensor1::zero(x.atoms,x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensor1 linmaps1(const Ptensor1& x){
    Ptensor1 R=Ptensor1::zero(x.atoms,2*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensor1 linmaps1(const Ptensor2& x){
    Ptensor1 R=Ptensor1::zero(x.atoms,5*x.nc,x.dev);
    x.add_linmaps_to(R);
    return R;
  }


  inline Ptensor2 linmaps2(const Ptensor0& x){
    Ptensor2 R=Ptensor2::zero(x.atoms,2*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensor2 linmaps2(const Ptensor1& x){
    Ptensor2 R=Ptensor2::zero(x.atoms,5*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensor2 linmaps2(const Ptensor2& x){
    Ptensor2 R=Ptensor2::zero(x.atoms,15*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

}


#endif 

