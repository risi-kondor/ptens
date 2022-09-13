#ifndef _EQUIVARIANT_MESSAGE_PASSING
#define _EQUIVARIANT_MESSAGE_PASSING

namespace ptens{

  void add_messages_0_to_1(Ptensor1& r, const Ptensor0& x, int offs=0){ // 1
    
    Atoms common=atoms.intersect(x.atoms);
    int k=common.size();
    vector<int> ix(atoms(common));
    vector<int> xix(x.atoms(common));

    

  }
    
  void add_messages_0_to_1_back(Ptensor0&  rconst Ptensor1& x, int offs=0) const{ // 1
  }

 

}

#endif 
