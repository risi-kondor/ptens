#ifndef _ptens_iipair
#define _ptens_iipair

namespace ptens{

  class iipair{
  public:

    int i0;
    int i1;
    
    iipair(const int _i0, const int _i1): i0(_i0), i1(_i1){}

    bool operator==(const iipair& x) const{
      return (i0==x.i0)&&(i1==x.i1);
    }

  };

}


namespace std{
  template<>
  struct hash<ptens::iipair>{
  public:
    size_t operator()(const ptens::iipair& sgntr) const{
      return (hash<int>()(sgntr.i0)<<1)^hash<int>()(sgntr.i1); 
    }
  };
}



#endif 
