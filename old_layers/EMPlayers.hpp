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
#ifndef _ptens_EMPlayers
#define _ptens_EMPlayers

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Hgraph.hpp"


namespace ptens{

  // 0 -> 0
  void add_msg(Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }

  // 0 -> 1
  void add_msg(Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }
    
  // 0 -> 2
  void add_msg(Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }


  // 1 -> 0
  void add_msg(Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }

  void add_msg_n(Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0_n(indices.first),indices.second,offs);
  }
  void add_msg_back_n(Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0_n(x.reduce0(indices.first,offs,r.nc),indices.second);
  }


  // 1 -> 1
  void add_msg(Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+nc);
  }
  void add_msg_back(Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+nc,nc),indices.second);
  }

  void add_msg_n(Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0_n(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+nc);
  }
  void add_msg_back_n(Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0_n(x.reduce0(indices.first,offs,nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+nc,nc),indices.second);
  }


  // 1 -> 2
  void add_msg(Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+2*nc);
  }
  void add_msg_back(Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+2*nc,nc),indices.second);
  }

  void add_msg_n(Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0_n(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+2*nc);
  }
  void add_msg_back_n(Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0_n(x.reduce0(indices.first,offs,nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+2*nc,nc),indices.second);
  }


  // 2 -> 0
  void add_msg(Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,2*nc),indices.second);
  }

  void add_msg_n(Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0_n(indices.first),indices.second,offs);
  }
  void add_msg_back_n(Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0_n(x.reduce0(indices.first,offs,2*nc),indices.second);
  }


  // 2 -> 1
  void add_msg(Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+2*nc);
  }
  void add_msg_back(Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,2*nc),indices.second); // !!
    r.broadcast1(x.reduce1(indices.first,offs+2*nc,3*nc),indices.second);
  }

  void add_msg_n(Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0_n(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1_n(indices.first),indices.second,offs+2*nc);
  }
  void add_msg_back_n(Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0_n(x.reduce0(indices.first,offs,2*nc),indices.second); // !!
    r.broadcast1_n(x.reduce1(indices.first,offs+2*nc,3*nc),indices.second);
  }


  // 2 -> 2
  void add_msg(Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+4*nc);
    r.broadcast2(x.reduce2(indices.first),indices.second,offs+13*nc);
  }
    
  void add_msg_back(Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,2*nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+4*nc,3*nc),indices.second);
    r.broadcast2(x.reduce2(indices.first,offs+13*nc,nc),indices.second);
  }
    
  void add_msg_n(Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0_n(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1_n(indices.first),indices.second,offs+4*nc);
    r.broadcast2(x.reduce2(indices.first),indices.second,offs+13*nc);
  }
    
  void add_msg_back_n(Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    if(G.is_empty()) return;
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0_n(x.reduce0(indices.first,offs,2*nc),indices.second);
    r.broadcast1_n(x.reduce1(indices.first,offs+4*nc,3*nc),indices.second);
    r.broadcast2(x.reduce2(indices.first,offs+13*nc,nc),indices.second);
  }
    


  // --------------------------------------------------------------------------------------------------------


  Ptensors1 unite1(const Ptensors0& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.merge(x.atoms),x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
  
  Ptensors1 unite1(const Ptensors1& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.merge(x.atoms),2*x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
  
  Ptensors1 unite1(const Ptensors2& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.merge(x.atoms),5*x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }

  
  Ptensors2 unite2(const Ptensors0& x, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(G.merge(x.atoms),2*x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
  
  Ptensors2 unite2(const Ptensors1& x, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(G.merge(x.atoms),5*x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
  
  Ptensors2 unite2(const Ptensors2& x, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(G.merge(x.atoms),15*x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
   


  Ptensors1 unite1_n(const Ptensors0& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.merge(x.atoms),x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
  
  Ptensors1 unite1_n(const Ptensors1& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.merge(x.atoms),2*x.nc,x.dev);
    add_msg_n(R,x,G);
    return R;
  }
  
  Ptensors1 unite1_n(const Ptensors2& x, const Hgraph& G){
    Ptensors1 R=Ptensors1::zero(G.merge(x.atoms),5*x.nc,x.dev);
    add_msg_n(R,x,G);
    return R;
  }

  
  Ptensors2 unite2_n(const Ptensors0& x, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(G.merge(x.atoms),2*x.nc,x.dev);
    add_msg(R,x,G);
    return R;
  }
  
  Ptensors2 unite2_n(const Ptensors1& x, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(G.merge(x.atoms),5*x.nc,x.dev);
    add_msg_n(R,x,G);
    return R;
  }
  
  Ptensors2 unite2_n(const Ptensors2& x, const Hgraph& G){
    Ptensors2 R=Ptensors2::zero(G.merge(x.atoms),15*x.nc,x.dev);
    add_msg_n(R,x,G);
    return R;
  }
   
}


#endif 
