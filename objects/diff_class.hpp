#ifndef _diff_class
#define _diff_class

#include "Cnine_base.hpp"
#include "loose_ptr.hpp"


namespace ptens{

  template<typename OBJ>
  class diff_class{
  public:

#ifdef WITH_FAKE_GRAD

    OBJ* grad=nullptr;


    void add_to_grad(const OBJ& x){
      if(grad) grad->add(x);
      else grad=new OBJ(x);
    }

    OBJ& get_grad(){
      if(!grad) grad=OBJ::new_zeros_like(static_cast<OBJ&>(*this));
      return *grad;
    }

    //const OBJ& get_grad() const{
    //if(!grad) grad=OBJ::new_zeros_like(static_cast<OBJ&>(*this));
    //return *grad;
    //}

    loose_ptr<OBJ> get_gradp(){
      if(!grad) grad=OBJ::new_zeros_like(static_cast<OBJ&>(*this));
      return grad;
    }

#endif 


  };

}

#endif
