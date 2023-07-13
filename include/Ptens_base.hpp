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

#define _PTENS_GEN_ASSERTS

#ifdef _PTENS_GEN_ASSERTS
#define PTENS_K_SAME(x) if(x.k!=k) throw std::invalid_argument("Ptens error in "+string(__PRETTY_FUNCTION__)+": reference domain size mismatch between "+to_string(k)+" and "+to_string(x.k)+".");
#define PTENS_CHANNELS(cond) if(!(cond)) throw std::invalid_argument("Ptens error in "+string(__PRETTY_FUNCTION__)+": channel mismatch.");
#else
#define PTENS_K_SAME(x)
#define PTENS_CHANNELS(cond)
#endif 

#define PTENS_ASSRT(condition) \
  if(!(condition)) throw std::runtime_error("Ptens error in "+string(__PRETTY_FUNCTION__)+": failed assertion "+#condition+".");


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


#endif 


