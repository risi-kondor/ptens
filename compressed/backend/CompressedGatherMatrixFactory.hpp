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

#ifndef _ptens_CompressedGatherMatrixFactory
#define _ptens_CompressedGatherMatrixFactory

#include "CompressedGatherMatrix.hpp"
#include "CompressedAtomsPackObj.hpp"


namespace ptens{


  template<int outk, int ink, int gatherk>
  class CompressedGatherMatrixFactory{
  public:


    static CompressedGatherMatrix gather_matrix(const LayerMap& map, const CompressedAtomsPack& out, 
      const CompressedAtomsPack& in){
      return make_or_cached(*map.obj,*out.obj,*in.obj);
    }


    static shared_ptr<CompressedGatherMatrixObj> make_or_cached(const LayerMapObj& map, 
      const CompressedAtomsPackObj& out, const CompressedAtomsPackObj& in){

      if(ptens_global::gather_matrix_cache.contains(map,out,in,9*outk+3*ink+gatherk))
	return ptens_global::gather_matrix_cache(map,out,in,9*outk+3*ink+gatherk);

      auto R=make(map,out,in);
      ptens_global::gather_matrix_cache.insert(map,out,in,9*outk+3*ink+gatherk,R);
      return R;
    }


    static shared_ptr<CompressedGatherMatrixObj> make(const LayerMapObj& map, 
      const CompressedAtomsPackObj& out, const CompressedAtomsPackObj& in){

      int outdim=pow(out.nvecs(),outk);
      int indim=pow(in.nvecs(),ink);
      auto R=new CompressedGatherMatrixObj(out.size(),in.size(),outdim,indim,map,0);

      map.for_each([&](const int i, const int j){
	  Atoms in_j=(*in.atoms)[j];
	  Atoms out_i=(*out.atoms)[i];
	  //Atoms common=out_i.intersect(in_j);
	  //int nix=common.size();
	  auto [out_common,in_common]=out_i.intersecting(in_j);

	  if(outk==0){
	    PTENS_ASSRT(gatherk==0);
	    auto r_tensor=reduction_tensor(ink,in.basis(j),in_common);
	    //R->block(i,j).add_einsum("akl->(al)k",r_tensor);
	    R->block(i,j).slice(0,0)+=r_tensor.slice(0,0).transp().fuse(0,1);
	  }else{
	    if(ink==0){
	      auto b_tensor=reduction_tensor(outk,out.basis(j),in_common);
	      //R->block(i,j).add_einsum("aij->(ij)a",r_tensor);
	      R->block(i,j).slice(1,0)+=b_tensor.slice(0,0).fuse(0,1);
	    }else{
	      auto r_tensor=reduction_tensor(ink,in.basis(j),in_common);
	      auto b_tensor=reduction_tensor(outk,out.basis(j),in_common);
	      //R->block(i,j).add_einsum("aij,akl->(ijl)k",b_tensor,r_tensor);
	      int rmult=r_tensor.dim(2);
	      if(rmult==0){
		R->block(i,j).view2().add_mprod(b_tensor.view3().fuse12().transp(),r_tensor.slice(2,0).view2());
	      }else{
		for(int i=0; i<rmult; i++)
		  R->block(i,j).split(0,rmult).slice(1,i).view2().
		    add_mprod(b_tensor.view3().fuse12().transp(),r_tensor.slice(2,i).view2());
	      }
	    }
	  }
	});

      return cnine::to_share(R);
    }


    static cnine::TensorView<float> reduction_tensor(const int order, const cnine::TensorView<float>& basis, 
      const vector<int>& common, const bool is_broadcast=false){
      int n=common.size();
      int channel_multiplier=1;
      if (order==2) channel_multiplier=vector<int>({2,3,1+is_broadcast})[gatherk];
      cnine::TensorView<float> R(cnine::dims(pow(common.size(),gatherk),pow(basis.dim(1),order),channel_multiplier));

      if(order==1){

	if constexpr(gatherk==0){
	  for(int i=0; i<common.size(); i++)
	    R.slice(2,0).row(0)+=basis.row(common[i]);
	}

	if constexpr(gatherk==1){
	  for(int i=0; i<common.size(); i++)
	    R.slice(2,0).row(i)=basis.row(common[i]);
	}

      }

      if(order==2){

	if constexpr(gatherk==0){
	  for(int i=0; i<common.size(); i++){
	    for(int j=0; j<common.size(); j++)
	      R.slice(2,0).row(0)+=kron(basis.row(common[i]),basis.row(common[j]));
	    R.slice(2,1).row(0)+=kron(basis.row(common[i]),basis.row(common[i]));
	  }
	}

	if constexpr(gatherk==1){
	  for(int i=0; i<common.size(); i++){
	    for(int j=0; j<common.size(); j++){
	      R.slice(2,0).row(i)=kron(basis.row(common[j]),basis.row(common[i]));
	      R.slice(2,1).row(i)=kron(basis.row(common[i]),basis.row(common[j]));
	    }
	    R.slice(2,2).row(i)=kron(basis.row(common[i]),basis.row(common[i]));
	  }
	}

	if constexpr(gatherk==2){
	  for(int i=0; i<common.size(); i++)
	    for(int j=0; j<common.size(); j++)
	      R.slice(2,0).row(i*n+j)=kron(basis.row(common[i]),basis.row(common[j]));
	  if(is_broadcast){
	    for(int i=0; i<common.size(); i++)
	      for(int j=0; j<common.size(); j++)
		R.slice(2,1).row(i*n+j)=kron(basis.row(common[j]),basis.row(common[i]));
	  }
	}

      }

      return R;
    }


  };


}

#endif 

  /*
    template<int ink, int gatherk>
    class CompressedReductionMatrix{

    };
    
    template<>
    class CompressedReductionMatrix<1,0>{
    public:
      static cnine::TensorView<float> matrix(const cnine::TensorView<float>& basis, const vector<int>& common){
	cnine::TensorView<float> R({basis.dim(1)});
	for(int i=0; i<common.size(); i++)
	  R+=basis.row(common[i]);
	return R;
      }
    };

    template<>
    class CompressedReductionMatrix<1,1>{
    public:
      static cnine::TensorView<float> matrix(const cnine::TensorView<float>& basis, const vector<int>& common){
	cnine::TensorView<float> R({(int)common.size(),basis.dim(1)});
	for(int i=0; i<common.size(); i++)
	  R.row(i)=basis.row(common[i]);
	return R;
      }
    };

    template<>
    class CompressedReductionMatrix<2,0>{
    public:
      static cnine::TensorView<float> matrix(const cnine::TensorView<float>& basis, const vector<int>& common){
	cnine::TensorView<float> R({(int)common.size(),basis.dim(1)});
	for(int i=0; i<common.size(); i++)
	  R.row(i)=basis.row(common[i]);
	return R;
      }
    };
  */
