#ifndef _AddMsgFunctions
#define _AddMsgFunctions

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
    r.broadcast0(x,r.atoms(common),offs);
  }
  void add_msg_back(Ptensor0& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    r.broadcast0(x.reduce0(offs,r.nc));
  }


  // 1 -> 0
  void add_msg(Ptensor0& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    r.broadcast0(x.reduce0(x.atoms(common)),offs);
  }

  void add_msg(Ptensor1& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+nc);
  }
    
  void add_msg(Ptensor2& r, const Ptensor1& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+2*nc);
  }



  void add_msg(Ptensor0& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    r.broadcast0(x.reduce0(x.atoms(common)),offs);
  }

  void add_msg(Ptensor1& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+2*nc);
  }

  void add_msg(Ptensor2& r, const Ptensor2& x, int offs=0){
    Atoms common=r.atoms.intersect(x.atoms);
    int nc=x.get_nc();
    vector<int> rix(r.atoms(common));
    vector<int> xix(x.atoms(common));
    r.broadcast0(x.reduce0(xix),rix,offs);
    r.broadcast1(x.reduce1(xix),rix,offs+4*nc);
    r.broadcast2(x.reduce2(xix),rix,offs+13*nc);
  }
    


  
}

#endif 


    //Atoms common=atoms.intersect(x.atoms);
    //int k=common.size();
    //vector<int> rix(r.atoms(common));
    //vector<int> xix(x.atoms(common));


  //void add_msg_back(Ptensor0&  r, const Ptensor1& x, int offs=0){
  //}
  
