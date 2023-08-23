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

#ifndef _ptens_LinmapLayers
#define _ptens_LinmapLayers

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"

#include "PtensLoggedTimer.hpp"
//extern ptens::PtensSession ptens_session;


namespace ptens{

  // 0 -> 0
  inline void add_linmaps(Ptensors0& r, const Ptensors0& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors0& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  // 0 -> 1
  inline void add_linmaps(Ptensors1& r, const Ptensors0& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors0& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  // 0 -> 2
  inline void add_linmaps(Ptensors2& r, const Ptensors0& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors0& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }


  // 1 -> 0
  inline void add_linmaps(Ptensors0& r, const Ptensors1& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors1& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  inline void add_linmaps_n(Ptensors0& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0_n(),offs);
  }
  inline void add_linmaps_back_n(Ptensors1& r, const Ptensors0& x, const int offs=0){
    r.broadcast0_n(x.reduce0(offs,r.nc));
  }


  // 1 -> 1
  inline void add_linmaps(Ptensors1& r, const Ptensors1& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+x.nc);
  }
  inline void add_linmaps_back(Ptensors1& r, const Ptensors1& x, const int offs=0){
    //LoggedTimer("add linmaps back ",x," -> ",r);
    r.broadcast0(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+r.nc,r.nc));
  }

  inline void add_linmaps_n(Ptensors1& r, const Ptensors1& x, const int offs=0){
    //LoggedTimer("add linmap_n ",x," -> ",r);
    r.broadcast0(x.reduce0_n(),offs);
    r.broadcast1(x.reduce1(),offs+x.nc);
  }
  inline void add_linmaps_back_n(Ptensors1& r, const Ptensors1& x, const int offs=0){
    //LoggedTimer("add linmaps_n back ",x," -> ",r);
    r.broadcast0_n(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+r.nc,r.nc));
  }


  // 1 -> 2
  inline void add_linmaps(Ptensors2& r, const Ptensors1& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+2*x.nc);
  }
  inline void add_linmaps_back(Ptensors1& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+2*r.nc,r.nc));
  }

  inline void add_linmaps_n(Ptensors2& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0_n(),offs);
    r.broadcast1(x.reduce1(),offs+2*x.nc);
  }
  inline void add_linmaps_back_n(Ptensors1& r, const Ptensors2& x, const int offs=0){
    r.broadcast0_n(x.reduce0(offs,r.nc));
    r.broadcast1(x.reduce1(offs+2*r.nc,r.nc));
  }



  // 2 -> 0
  inline void add_linmaps(Ptensors0& r, const Ptensors2& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
  }
  inline void add_linmaps_back(Ptensors2& r, const Ptensors0& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,2*r.nc)); // changed
  }

  inline void add_linmaps_n(Ptensors0& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0_n(),offs);
  }
  inline void add_linmaps_back_n(Ptensors2& r, const Ptensors0& x, const int offs=0){
    r.broadcast0_n(x.reduce0(offs,2*r.nc)); // changed
  }


  // 2 -> 1
  inline void add_linmaps(Ptensors1& r, const Ptensors2& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+2*x.nc);
  }
  inline void add_linmaps_back(Ptensors2& r, const Ptensors1& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,2*r.nc)); // changed
    r.broadcast1(x.reduce1(offs+2*r.nc,3*r.nc)); // changed
  }

  inline void add_linmaps_n(Ptensors1& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0_n(),offs);
    r.broadcast1(x.reduce1_n(),offs+2*x.nc);
  }
  inline void add_linmaps_back_n(Ptensors2& r, const Ptensors1& x, const int offs=0){
    r.broadcast0_n(x.reduce0(offs,2*r.nc)); // changed
    r.broadcast1_n(x.reduce1(offs+2*r.nc,3*r.nc)); // changed
  }


  // 2 -> 2
  inline void add_linmaps(Ptensors2& r, const Ptensors2& x, const int offs=0){
    //LoggedTimer("add linmaps ",x," -> ",r);
    r.broadcast0(x.reduce0(),offs);
    r.broadcast1(x.reduce1(),offs+4*x.nc);
    r.broadcast2(x.reduce2(),offs+13*x.nc);
  }
  inline void add_linmaps_back(Ptensors2& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0(offs,2*r.nc)); // changed 
    r.broadcast1(x.reduce1(offs+4*r.nc,3*r.nc)); // changed 
    r.broadcast2(x.reduce2(offs+13*r.nc,r.nc));
  }

  inline void add_linmaps_n(Ptensors2& r, const Ptensors2& x, const int offs=0){
    r.broadcast0(x.reduce0_n(),offs);
    r.broadcast1(x.reduce1_n(),offs+4*x.nc);
    r.broadcast2(x.reduce2(),offs+13*x.nc);
  }
  inline void add_linmaps_back_n(Ptensors2& r, const Ptensors2& x, const int offs=0){
    r.broadcast0_n(x.reduce0(offs,2*r.nc)); // changed 
    r.broadcast1_n(x.reduce1(offs+4*r.nc,3*r.nc)); // changed 
    r.broadcast2(x.reduce2(offs+13*r.nc,r.nc));
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



  inline Ptensors0 linmaps0_n(const Ptensors1& x){
    Ptensors0 R=Ptensors0::zero(x.atoms,x.nc,x.dev);
    add_linmaps_n(R,x);
    return R;
  }

  inline Ptensors1 linmaps1_n(const Ptensors1& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,2*x.nc,x.dev);
    add_linmaps_n(R,x);
    return R;
  }

  inline Ptensors2 linmaps2_n(const Ptensors1& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,5*x.nc,x.dev);
    add_linmaps_n(R,x);
    return R;
  }


  inline Ptensors0 linmaps0_n(const Ptensors2& x){
    Ptensors0 R=Ptensors0::zero(x.atoms,2*x.nc,x.dev);
    add_linmaps_n(R,x);
    return R;
  }

  inline Ptensors1 linmaps1_n(const Ptensors2& x){
    Ptensors1 R=Ptensors1::zero(x.atoms,5*x.nc,x.dev);
    add_linmaps_n(R,x);
    return R;
  }

  inline Ptensors2 linmaps2_n(const Ptensors2& x){
    Ptensors2 R=Ptensors2::zero(x.atoms,15*x.nc,x.dev);
    add_linmaps_n(R,x);
    return R;
  }

}

#endif 
