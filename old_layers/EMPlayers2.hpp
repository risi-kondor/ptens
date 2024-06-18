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

//#include "Ptensors0.hpp"
//#include "Ptensors1.hpp"
//#include "Ptensors2.hpp"
//#include "Hgraph.hpp"
#include "flog.hpp"


namespace ptens{

  template<typename SRC, typename DEST>
  void emp00(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    cnine::flog timer("ptens::emp00");
    r.broadcast0(x.reduce0(map.in()),map.out(),0);
  }

  template<typename SRC, typename DEST>
  void emp01(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    cnine::flog timer("ptens::emp01");
    r.broadcast0(x.reduce0(map.in()),map.out(),0);
  }

  template<typename SRC, typename DEST>
  void emp10(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    cnine::flog timer("ptens::emp10");
    r.broadcast0(x.reduce0(map.in()),map.out(),0);
  }

  template<typename SRC, typename DEST>
  void emp11(DEST& r, const SRC& x, const TransferMap& map){
    int nc=x.get_nc();
    if(!map.is_graded()){
      if(map.is_empty()) return;
      cnine::flog timer("ptens::emp11");
      r.broadcast0(x.reduce0(map.in()),map.out(),0);
      r.broadcast1(x.reduce1(map.in()),map.out(),nc);
    }else{
      cnine::flog timer("ptens::emp11G");
      for(auto& p:map.obj->graded_maps){
	auto& map=*p.second;
	r.broadcast0(x.reduce0(map.in),map.out,*map.bmap,0);
	r.broadcast1(x.reduce1(map.in),map.out,*map.bmap,nc);
      }
    }
  }

  template<typename SRC, typename DEST>
  void emp11_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    cnine::flog timer("ptens::emp11_back");
    r.reduce0_back(x.broadcast0_back(map.in(),0,nc),map.out());
    r.reduce1_back(x.broadcast1_back(map.in(),nc,nc),map.out());
  }


  
  template<typename SRC, typename DEST>
  void emp02(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    r.broadcast0(x.reduce0(map.in()),map.out());
  }

  template<typename SRC, typename DEST>
  void emp02_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    r.reduce0_back(x.broadcast0_back(map.in(),0,nc),map.out());
  }

  template<typename SRC, typename DEST>
  void emp12(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    r.broadcast0(x.reduce0(map.in()),map.out(),0);
    r.broadcast1(x.reduce1(map.in()),map.out(),2*nc);
  }

  template<typename SRC, typename DEST>
  void emp12_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    r.reduce0_back(x.broadcast0_back(map.in(),0,nc),map.out());
    r.reduce1_back(x.broadcast1_back(map.in(),2*nc,nc),map.out());
  }

  template<typename SRC, typename DEST>
  void emp22(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    r.broadcast0(x.reduce0(map.in()),map.out(),0);
    r.broadcast1(x.reduce1(map.in()),map.out(),4*nc);
    r.broadcast2(x.reduce2(map.in()),map.out(),13*nc);
  }

  template<typename SRC, typename DEST>
  void emp22_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    r.reduce0_back(x.broadcast0_back(map.in(),0,2*nc),map.out());
    r.reduce1_back(x.broadcast1_back(map.in(),4*nc,3*nc),map.out());
    r.reduce2_back(x.broadcast2_back(map.in(),13*nc,nc),map.out());
  }

  template<typename SRC, typename DEST>
  void emp20(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    r.broadcast0(x.reduce0(map.in()),map.out(),0);
  }

  template<typename SRC, typename DEST>
  void emp20_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    r.reduce0_back(x.broadcast0_back(map.in(),0,2*nc),map.out());
  }

  template<typename SRC, typename DEST>
  void emp21(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=x.get_nc();
    r.broadcast0(x.reduce0(map.in()),map.out(),0); // added 0 
    r.broadcast1(x.reduce1(map.in()),map.out(),2*nc);
  }

  template<typename SRC, typename DEST>
  void emp21_back(DEST& r, const SRC& x, const TransferMap& map){
    if(map.is_empty()) return;
    int nc=r.get_nc();
    r.reduce0_back(x.broadcast0_back(map.in(),0,2*nc),map.out());
    r.reduce1_back(x.broadcast1_back(map.in(),2*nc,3*nc),map.out());
  }


}

#endif 


//auto [map0,map1]=map.intersects(x.atoms,r.atoms);
