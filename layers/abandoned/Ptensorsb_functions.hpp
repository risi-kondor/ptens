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
 *
 */

#ifndef _ptens_Ptensorsb_functions
#define _ptens_Ptensorsb_functions

#include "Ptensors0b.hpp"
#include "Ptensors1b.hpp"
#include "Ptensors2b.hpp"
#include "SubgraphLayer0b.hpp"
#include "SubgraphLayer1b.hpp"
#include "SubgraphLayer2b.hpp"


namespace ptens{


  // ---- Linmaps to new Ptensor layers ---------------------------------------------------------------------

  //template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
  //inline Ptensors0b<float> linmaps0(const SOURCE& x){
  //Ptensors0b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
  //R.add_linmaps(x);
  //return R;
  //}
  /*
  template<typename SOURCE>
  inline Ptensors1b<float> linmaps1(const SOURCE& x){
    Ptensors1b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline Ptensors2b<float> linmaps2(const SOURCE& x){
    Ptensors2b<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }
  */

  // ---- Gathering to new Ptensor layer --------------------------------------------------------------------


  /*
  template<typename SOURCE>
  Ptensors0b<float> gather0(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,1,2})[x.getk()];
    Ptensors0b<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }

  template<typename SOURCE>
  Ptensors1b<float> gather1(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({1,2,5})[x.getk()];
    Ptensors0b<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }

  template<typename SOURCE>
  Ptensors2b<float> gather2(const SOURCE& x, const AtomsPack& a){
    int nc=x.get_nc()*vector<int>({2,5,15})[x.getk()];
    Ptensors2b<float> R(a,nc,x.get_dev());
    R.add_gather(x);
    return R;
  }
  */



}

#endif 


  /* these do not work 
  template<typename TYPE>
  inline Ptensors0b<TYPE> linmaps0(const Ptensorsb<TYPE>& x){
    Ptensors0b<TYPE> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensors1b<TYPE> linmaps1(const Ptensorsb<TYPE>& x){
    Ptensors1b<float> R(x.atoms,x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename TYPE>
  inline Ptensors2b<TYPE> linmaps2(const Ptensorsb<TYPE>& x){
    Ptensors2b<float> R(x.atoms,x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }
  */


  /*
  template<typename SOURCE>
  inline SubgraphLayer0b<float> sglinmaps0(const SOURCE& x){
    SubgraphLayer0b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer1b<float> sglinmaps1(const SOURCE& x){
    SubgraphLayer1b<float> R(x.get_atoms(),x.get_nc()*vector<int>({1,2,5})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }

  template<typename SOURCE>
  inline SubgraphLayer2b<float> sglinmaps2(const SOURCE& x){
    SubgraphLayer2b<float> R(x.get_atoms(),x.get_nc()*vector<int>({2,5,15})[x.getk()],x.get_dev());
    R.add_linmaps(x);
    return R;
  }
  */
