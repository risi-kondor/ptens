#ifndef _ptens_EMPlayers
#define _ptens_EMPlayers

#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"
#include "Hgraph.hpp"


namespace ptens{

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
   
}


#endif 
