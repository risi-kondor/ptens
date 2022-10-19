#ifndef _Ptensor_subpack
#define _Ptensor_subpack

#include "AtomsPack.hpp"
#include "Ptensor1.hpp"
#include "Cgraph.hpp"


namespace ptens{

  //template<>
  class Ptensor_subpack: public cnine::RtensorA{
  public:
    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;

    AtomsPack atoms;


  public:

    using rtensor::rtensor;


  public: // ---- Copying -------------------------------------------------------------------------------------


    Ptensor_subpack(const Ptensor_subpack& x):
      rtensor(x),
      atoms(x.atoms){
    }


    Ptensor_subpack(Ptensor_subpack&& x):
      rtensor(std::move(x)),
      atoms(std::move(x.atoms)){
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    Ptensor_subpack(const AtomsPack& _atoms, const rtensor& x):
      rtensor(x), 
      atoms(_atoms){
      assert(atoms.getn()==getn());
      assert(atoms.getk()==getk());
    }

   Ptensor_subpack(const AtomsPack& _atoms, rtensor&& x):
     rtensor(std::move(x)), 
     atoms(_atoms){
      assert(atoms.getn()==getn());
      assert(atoms.getk()==getk());
    }




  };

}

#endif
