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
#ifndef _MsgFunctions
#define _MsgFunctions

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"


// ---- Individual Ptensors ----------------------------------------------------------------------------------

namespace ptens{


  // 0 -> 0
  void add_msg(Ptensor0& r, const Ptensor0& x, int offs=0){
    r.broadcast0(x,offs);
  }
  void add_msg_back(Ptensor0& r, const Ptensor0& x, int offs=0){
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  // 0 -> 1
  void add_msg(Ptensor1& r, const Ptensor0& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    r.broadcast0(x,r.atoms(common),offs);
  }
  void add_msg_back(Ptensor0& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    r.broadcast0(x.reduce0(offs,r.nc));
  }
    
  // 0 -> 2
  void add_msg(Ptensor2& r, const Ptensor0& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x,rix,offs);
  }
  void add_msg_back(Ptensor0& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix,offs,r.nc));
  }


  // 1 -> 0
  void add_msg(Ptensor0& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),offs);
  }
  void add_msg_back(Ptensor1& r, const Ptensor0& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(offs,r.nc));
  }

  // 1 -> 1
  void add_msg(Ptensor1& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+nc);
  }
  void add_msg_back(Ptensor1& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=r.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix,offs,nc),rix);
    r.broadcast1(x.reduce1(xix,offs+nc,nc),rix);
  }


  // 1 -> 2
  void add_msg(Ptensor2& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+2*nc);
  }
  void add_msg_back(Ptensor1& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=r.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix,offs,nc),rix);
    r.broadcast1(x.reduce1(xix,offs+2*nc,nc),rix);
  }


  // 2 -> 0
  void add_msg(Ptensor0& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),offs);
  }
  void add_msg_back(Ptensor2& r, const Ptensor0& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=r.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix,offs,nc),rix);
  }

  // 2 -> 1
  void add_msg(Ptensor1& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+2*nc);
  }
  void add_msg_back(Ptensor2& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=r.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix,offs,nc),rix);
    r.broadcast1(x.reduce1(xix,offs+2*nc,nc),rix);
  }

  // 2 -> 2
  void add_msg(Ptensor2& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+4*nc);
    r.broadcast2(x.reduce2(xix),rix,offs+13*nc);
  }
    
  void add_msg_back(Ptensor2& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=r.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix,offs,nc),rix);
    r.broadcast1(x.reduce1(xix,offs+4*nc,nc),rix);
    r.broadcast2(x.reduce2(xix,offs+13*nc,nc),rix);
  }


  Ptensor0& operator>>(const Ptensor0& x, Ptensor0& r) {add_msg(r,x); return r;}
  Ptensor0& operator>>(const Ptensor1& x, Ptensor0& r) {add_msg(r,x); return r;}
  Ptensor0& operator>>(const Ptensor2& x, Ptensor0& r) {add_msg(r,x); return r;}
  Ptensor1& operator>>(const Ptensor0& x, Ptensor1& r) {add_msg(r,x); return r;}
  Ptensor1& operator>>(const Ptensor1& x, Ptensor1& r) {add_msg(r,x); return r;}
  Ptensor1& operator>>(const Ptensor2& x, Ptensor1& r) {add_msg(r,x); return r;}
  Ptensor2& operator>>(const Ptensor0& x, Ptensor2& r) {add_msg(r,x); return r;}
  Ptensor2& operator>>(const Ptensor1& x, Ptensor2& r) {add_msg(r,x); return r;}
  Ptensor2& operator>>(const Ptensor2& x, Ptensor2& r) {add_msg(r,x); return r;}
    
}




#endif 

