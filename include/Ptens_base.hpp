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

#endif 
