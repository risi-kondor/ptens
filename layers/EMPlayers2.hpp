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


namespace ptens{

  template<typename SRC, typename DEST>
  void emp00(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
  }

  template<typename SRC, typename DEST>
  void emp01(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
  }

  template<typename SRC, typename DEST>
  void emp10(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(map0),map1);
  }

  template<typename SRC, typename DEST>
  void emp11(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    auto [map0,map1]=map.intersects(x.atoms,r.atoms);
    r.broadcast1(x.reduce1(map0),map1);
  }

  
}

#endif 
