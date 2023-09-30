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

#ifndef _ptens_EMPlayers2
#define _ptens_EMPlayers2

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Hgraph.hpp"
#include "ftimer.hpp"


namespace ptens{

  template<typename SRC, typename DEST>
  void emp00(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1,0);
  }

  template<typename SRC, typename DEST>
  void emp01(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1,0);
  }

  template<typename SRC, typename DEST>
  void emp10(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1,0);
  }

  template<typename SRC, typename DEST>
  void emp11(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    cnine::ftimer timer("ptens::emp11");
    r.broadcast0(x.reduce0(map0),map1,0);
    r.broadcast1(x.reduce1(map0),map1,nc);
  }

  template<typename SRC, typename DEST>
  void emp11_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    cnine::ftimer timer("ptens::emp11_back");
    r.reduce0_back(x.broadcast0_back(map0,0,nc),map1);
    r.reduce1_back(x.broadcast1_back(map0,nc,nc),map1);
  }


  
  template<typename SRC, typename DEST>
  void emp02(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
  }

  template<typename SRC, typename DEST>
  void emp02_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.reduce0_back(x.broadcast0_back(map0,0,nc),map1);
  }

  template<typename SRC, typename DEST>
  void emp12(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
    r.broadcast1(x.reduce1(map0),map1,2*nc);
  }

  template<typename SRC, typename DEST>
  void emp12_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.reduce0_back(x.broadcast0_back(map0,0,nc),map1);
    r.reduce1_back(x.broadcast1_back(map0,2*nc,nc),map1);
  }

  template<typename SRC, typename DEST>
  void emp22(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
    r.broadcast1(x.reduce1(map0),map1,4*nc);
    r.broadcast2(x.reduce2(map0),map1,13*nc);
  }

  template<typename SRC, typename DEST>
  void emp22_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.reduce0_back(x.broadcast0_back(map0,0,2*nc),map1);
    r.reduce1_back(x.broadcast1_back(map0,4*nc,3*nc),map1);
    r.reduce2_back(x.broadcast2_back(map0,13*nc,nc),map1);
  }

  template<typename SRC, typename DEST>
  void emp20(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
  }

  template<typename SRC, typename DEST>
  void emp20_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.reduce0_back(x.broadcast0_back(map0,0,2*nc),map1);
  }

  template<typename SRC, typename DEST>
  void emp21(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
    r.broadcast1(x.reduce1(map0),map1,2*nc);
  }

  template<typename SRC, typename DEST>
  void emp21_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.reduce0_back(x.broadcast0_back(map0,0,2*nc),map1);
    r.reduce1_back(x.broadcast1_back(map0,2*nc,3*nc),map1);
  }




}

#endif 
