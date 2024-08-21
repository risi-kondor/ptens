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
 */

#ifndef _Ptens_base
#define _Ptens_base

#include "Cnine_base.hpp"
#include <string>
#include "TensorView.hpp"

#define _PTENS_GEN_ASSERTS

//#define PTENS_K_SAME(x) if(x.k!=k) throw std::invalid_argument("Ptens error in "+string(__PRETTY_FUNCTION__)+": reference domain size mismatch between "+to_string(getk())+" and "+to_string(x.k)+".");

#ifdef _PTENS_GEN_ASSERTS
#define PTENS_K_SAME(x) 
#define PTENS_CHANNELS(cond) if(!(cond)) throw std::invalid_argument("Ptens error in "+string(__PRETTY_FUNCTION__)+": channel mismatch.");
#else
#define PTENS_K_SAME(x)
#define PTENS_CHANNELS(cond)
#endif 

#define PTENS_ASSRT(condition) \
  if(!(condition)) throw std::runtime_error("Ptens error in "+string(__PRETTY_FUNCTION__)+": failed assertion "+#condition+".");

#define PTENS_UNIMPL() printf("Ptens error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);

#define PTENS_DEPRECATED() printf("Ptens warning: function \"%s\" deprecated.\n",__PRETTY_FUNCTION__);


// ---- Copy, assign and convert warnings --------------------------------------------------------------------


#ifdef PTENS_COPY_WARNINGS
#define PTENS_COPY_WARNING() cout<<"\e[1mptens:\e[0m "<<classname()<<" copied."<<endl;
#else 
#define PTENS_COPY_WARNING()
#endif 

#ifdef PTENS_ASSIGN_WARNINGS
#define PTENS_ASSIGN_WARNING() cout<<"\e[1mptens:\e[0m "<<classname()<<" assigned."<<endl;
#else
#define PTENS_ASSIGN_WARNING() 
#endif

#ifdef PTENS_MOVE_WARNINGS
#define PTENS_MOVE_WARNING() cout<<"\e[1mptens:\e[0m "<<classname()<<" moved."<<endl;
#else 
#define PTENS_MOVE_WARNING()
#endif 


// ---- Other -------------------------------------------------------------------------------------------------


#define PTENS_CPUONLY() if(dev!=0) {throw std::runtime_error("Ptens error: no CUDA code for "+string(__PRETTY_FUNCTION__)+".\n");}


// ---- Template decalarations -------------------------------------------------------------------------------

namespace ptens{

  //template<typename DUMMY> class PtensorsJig0;
  //template<typename DUMMY> class PtensorsJig1;
  //template<typename DUMMY> class PtensorsJig2;
  template<typename TYPE>
  using PtensTensor=cnine::TensorView<TYPE>;

  template<typename TYPE> class Ptensor0;
  template<typename TYPE> class Ptensor1;
  template<typename TYPE> class Ptensor2;

  template<typename TYPE> class Ptensors0;
  template<typename TYPE> class Ptensors1;
  template<typename TYPE> class Ptensors2;

  template<typename TYPE> class BatchedPtensors0;
  template<typename TYPE> class BatchedPtensors1;
  template<typename TYPE> class BatchedPtensors2;

  namespace ptens_global{};

}

#endif 


