#ifndef _ptens_LinMaps
#define _ptens_LinMaps

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"


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


  // ---- Ptensors layers -------------------------------------------------------------------------------------


  // 0 -> 0
  inline void add_linmaps(Ptensors0& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors0& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  // 0 -> 1
  inline void add_linmaps(Ptensors1& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors0& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  // 0 -> 2
  inline void add_linmaps(Ptensors2& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors0& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }


  // 1 -> 0
  inline void add_linmaps(Ptensors0& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors1& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }


  // 1 -> 1
  inline void add_linmaps(Ptensors1& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+x.nc);
  }
  inline void add_linmaps_back(Ptensors1& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+r.nc,r.nc));
  }


  // 1 -> 2
  inline void add_linmaps(Ptensors2& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+2*x.nc);
  }
  inline void add_linmaps_back(Ptensors1& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+2*r.nc,r.nc));
  }



  // 2 -> 0
  inline void add_linmaps(Ptensors0& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors2& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }


  // 2 -> 1
  inline void add_linmaps(Ptensors1& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+2*x.nc);
  }
  inline void add_linmaps_back(Ptensors2& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+2*r.nc,r.nc));
  }


  // 2 -> 2
  inline void add_linmaps(Ptensors2& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+4*x.nc);
    r.broadcast2(x.reduce2(),offs+13*x.nc);
  }
  inline void add_linmaps_back(Ptensors2& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+4*r.nc,r.nc));
    r.broadcast2(x.reduce2(offs+13*x.nc,r.nc));
  }


  inline Ptensors0 linmaps0(const Ptensors0& x){
    Ptensors0 R=Ptensors0::zero(x.atoms,x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }

  inline Ptensors1 linmaps1(const Ptensors0& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }

  inline Ptensors2 linmaps2(const Ptensors0& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,2*x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }


  inline Ptensors0 linmaps0(const Ptensors1& x){
    Ptensors0 R=Ptensors0::zero(x.atoms,x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }

  inline Ptensors1 linmaps1(const Ptensors1& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,2*x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }

  inline Ptensors2 linmaps2(const Ptensors1& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,5*x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }


  inline Ptensors0 linmaps0(const Ptensors2& x){
    Ptensors0 R=Ptensors0::zero(x.atoms,2*x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }

  inline Ptensors1 linmaps1(const Ptensors2& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,5*x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }

  inline Ptensors2 linmaps2(const Ptensors2& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,15*x.nc,x.dev);
    add_linmaps(R,x);
    return R;
  }



}


#endif 

  /*
  inline Ptensors1 linmaps1(const Ptensors0& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensors1 linmaps1(const Ptensors1& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,2*x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }


  inline Ptensors2 linmaps2(const Ptensors0& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,x.nc,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensors2 linmaps2(const Ptensors1& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,x.nc*3,x.dev);
    R.add_linmaps(x);
    return R;
  }

  inline Ptensors2 linmaps2(const Ptensors2& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,x.nc*15,x.dev);
    //R.add_linmaps(x);
    return R;
  }
  */
