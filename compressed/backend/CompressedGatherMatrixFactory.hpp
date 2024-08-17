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

    template<int ink, int gatherk>
    class CompressedReductionMatrix{

    };
    
    template<>
    class CompressedReductionMatrix<1,0>{
    public:
      static cnine::TensorView<float> matrix(const cnine::TensorView<float>& basis, const vector<int>& common){
	cnine::TensorView<float> R({basis.dim(1)});
	for(int i=0; i<common.size(); i++)
	  R+==basis.row(common[i]);
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



  template<int outk, int ink, int gatherk>
  class CompressedGatherMatrixFactory{
  public:

    static CompressedGatherMatrix gather_matrix(const LayerMap& map, const CompressedAtomsPack& out, 
      const CompressedAtomsPack& in){
      return make_or_cached(*map.obj,*out.obj,*in.obj); //,outk,ink,gatherk);
    }


    static shared_ptr<CompressedGatherMatrixObj> make_or_cached(const LayerMapObj& map, 
      const CompressedAtomsPackObj& out, const CompressedAtomsPackObj& in){
      //const int outk=0, const int ink=0, const int gatherk=0){

      if(ptens_global::gather_matrix_cache.contains(map,out,in,9*outk+3*ink+gatherk))
	return ptens_global::gather_matrix_cache(map,out,in,9*outk+3*ink+gatherk);

      auto R=make(map,out,in);//,outk,ink,gatherk);
      ptens_global::gather_matrix_cache.insert(map,out,in,9*outk+3*ink+gatherk,R);
      return R;
    }


    static shared_ptr<CompressedGatherMatrixObj> make(const LayerMapObj& map, 
      const CompressedAtomsPackObj& out, const CompressedAtomsPackObj& in){
      //const int outk=0, const int ink=0, const int gatherk=0){

      int outdim=pow(out.nvecs(),outk);
      int indim=pow(in.nvecs(),ink);
      auto R=new CompressedGatherMatrixObj(out.size(),in.size(),outdim,indim,map,0);

      int c=0;
      int toffset=0;
      map.for_each([&](const int i, const int j){
	  Atoms in_j=(*in.atoms)[j];
	  Atoms out_i=(*out.atoms)[i];
	  //Atoms common=out_i.intersect(in_j);
	  //int nix=common.size();
	  auto [out_common,in_common]=out_i.intersecting(in_j);

	  R->block(i,j)+=CompressedReductionMatrix<ink,gatherk>::matrix(in.basis(j),in_common);

	  /*
	  if(outk==0) out_pack->set(c,toffset,nix,out.row_offset0(i),out.nrows0(i),out[i](common));
	  if(outk==1) out_pack->set(c,toffset,nix,out.row_offset1(i),out.size_of(i),out[i](common));
	  if(outk==2) out_pack->set(c,toffset,nix,out.row_offset2(i),out.size_of(i),out[i](common));

	  if(ink==0) in_pack->set(c,toffset,nix,in.row_offset0(j),in.nrows0(j),in[j](common));
	  if(ink==1) in_pack->set(c,toffset,nix,in.row_offset1(j),in.size_of(j),in[j](common));
	  if(ink==2) in_pack->set(c,toffset,nix,in.row_offset2(j),in.size_of(j),in[j](common));

	  out_lists.push_back((*out_pack)(c,2),c);
	  in_lists.push_back((*in_pack)(c,2),c);

	  if(gatherk==0) toffset+=1;
	  if(gatherk==1) toffset+=nix;
	  if(gatherk==2) toffset+=nix*nix;
	  */
	  c++;
	});


      return cnine::to_share(R);
    }

     
    
  };


    

}


#endif 
