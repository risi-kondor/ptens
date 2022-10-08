#ifndef _loose_ptr
#define _loose_ptr

#include "Cnine_base.hpp"

namespace ptens{

  template<typename OBJ>
  class loose_ptr{
  public:

    OBJ* obj;

    loose_ptr(OBJ* _obj): obj(_obj){}
    
    operator OBJ&() const {return *obj;}

    OBJ& operator*() const{return *obj;}

    OBJ* operator->() const{return obj;}

  public:


  };

}

#endif
