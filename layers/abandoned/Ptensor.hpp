#ifndef _ptens_Ptensor
#define _ptens_Ptensor

#include "Atoms.hpp"
#include "RtensorObj.hpp"

#define PTENSOR_PTENSOR_IMPL cnine::RtensorObj


namespace ptens{

  class Ptensor: public PTENSOR_PTENSOR_IMPL{
  public:

    Atoms atoms;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor(const int _k, const Atoms& _atoms, const FILLTYPE& dummy, const int _dev=0):
      PTENSOR_PTENSOR_IMPL(vector<int>(_k,_atoms.size()),dummy,_dev),
      atoms(_atoms){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor raw(const int _k, const Atoms& _atoms, const int _dev=0){
	return Ptensor(_k,_atoms,cnine::fill_raw(),_dev);}

    static Ptensor zero(const int _k, const Atoms& _atoms, const int _dev=0){
	return Ptensor(_k,_atoms,cnine::fill_zero(),_dev);}

    static Ptensor gaussian(const int _k, const Atoms& _atoms, const int _dev=0){
	return Ptensor(_k,_atoms,cnine::fill_gaussian(),_dev);}

    


  };


}


#endif 
