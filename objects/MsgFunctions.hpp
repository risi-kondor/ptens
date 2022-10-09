#ifndef _MsgFunctions
#define _MsgFunctions

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Hgraph.hpp"


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
    

  // --------------------------------------------------------------------------------------------------------


  // 0 -> 0
  void add_msg(Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors0& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }

  // 0 -> 1
  void add_msg(Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }
    
  // 0 -> 2
  void add_msg(Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }


  // 1 -> 0
  void add_msg(Ptensors0& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors1& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,r.nc),indices.second);
  }

  // 1 -> 1
  void add_msg(Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+nc);
  }
  void add_msg_back(Ptensors1& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+nc,nc),indices.second);
  }


  // 1 -> 2
  void add_msg(Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+2*nc);
  }
  void add_msg_back(Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+2*nc,nc),indices.second);
  }


  // 2 -> 0
  void add_msg(Ptensors0& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
  }
  void add_msg_back(Ptensors2& r, const Ptensors0& x, const Hgraph& G, int offs=0){
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,2*nc),indices.second);
  }

  // 2 -> 1
  void add_msg(Ptensors1& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs); // !!
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+2*nc);
  }
  void add_msg_back(Ptensors2& r, const Ptensors1& x, const Hgraph& G, int offs=0){
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,2*nc),indices.second); // !!
    r.broadcast1(x.reduce1(indices.first,offs+2*nc,3*nc),indices.second);
  }

  // 2 -> 2
  void add_msg(Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    int nc=x.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first),indices.second,offs);
    r.broadcast1(x.reduce1(indices.first),indices.second,offs+4*nc);
    r.broadcast2(x.reduce2(indices.first),indices.second,offs+13*nc);
  }
    
  void add_msg_back(Ptensors2& r, const Ptensors2& x, const Hgraph& G, int offs=0){
    int nc=r.get_nc();
    auto indices=G.intersects(x.atoms,r.atoms);
    r.broadcast0(x.reduce0(indices.first,offs,2*nc),indices.second);
    r.broadcast1(x.reduce1(indices.first,offs+4*nc,3*nc),indices.second);
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


  // --------------------------------------------------------------------------------------------------------


  void add_gather(Ptensors0& r, const Ptensors0& x, const Hgraph& G){
    PTENS_ASSRT(G.n==r.size());
    PTENS_ASSRT(G.m==x.size());
    G.forall_edges([&](const int i, const int j, const float v){
	r.view_of_tensor(i).add(x.view_of_tensor(j),v);
      });
  }

  Ptensors0 gather(const Ptensors0& x, const Hgraph& G){
    Ptensors0 R=Ptensors0::zero(G.n,x.get_nc());
    add_gather(R,x,G);
    return R;
  }
   
}

#endif 

