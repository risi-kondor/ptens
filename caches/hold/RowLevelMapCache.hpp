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

#ifndef _ptens_RowLevelMapCache
#define _ptens_RowLevelMapCache

#include "ptr_triple_indexed_cache.hpp"
//#include "AtomsPack.hpp"
//#include "AtomsPackObj.hpp"
#include "AtomsPackTag.hpp"
#include "PtensorMapObj.hpp"
#include "RowLevelMap.hpp"


namespace ptens{

  typedef AtomsPackObj DUMMYC;


  namespace ptens_global{
    extern bool cache_rmaps;
  }


  class RowLevelMapCache: 
    public cnine::ptr_triple_indexed_cache<AtomsPackTagObj,AtomsPackTagObj,PtensorMapObj,shared_ptr<RowLevelMap> >{
  public:

    typedef std::tuple<AtomsPackTagObj*,AtomsPackTagObj*,PtensorMapObj*> KEYS;
    typedef shared_ptr<RowLevelMap> OBJ;
    typedef cnine::ptr_triple_indexed_cache<AtomsPackTagObj,AtomsPackTagObj,PtensorMapObj,shared_ptr<RowLevelMap> > BASE;

    typedef cnine::Gdims Gdims;
    //typedef cnine::GatherMapProgram GatherMapProgram;
    typedef cnine::TensorProgram<cnine::GatherRows,cnine::GatherMapB> GatherMapProgram;


    RowLevelMapCache(){}


  public: // ---- Access ------------------------------------------------------------------------------------------


    template<typename OUT_TAG, typename IN_TAG>
    shared_ptr<RowLevelMap> operator()(const OUT_TAG& out, const IN_TAG& in, const shared_ptr<PtensorMapObj>& map){
      auto out_p=out.obj.get();
      auto in_p=in.obj.get();
      auto p=make_tuple(out_p,in_p,map.get());
      auto it=find(p);
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      auto r=std::make_shared<RowLevelMap>(mmap(out.obj->get_atoms(),in.obj->get_atoms(),*map));
      //auto r=shared_ptr<RowLevelMap>(new RowLevelMap(mmap(out.obj->get_atoms(),in.obj->get_atoms(),*map)));
      BASE::insert(out_p,in_p,map.get(),r);
      return r;
    }
    

  private: // ---- Zeroth order ------------------------------------------------------------------------------------


    // 0 <- 0
    GatherMapProgram mmap(const AtomsPackObj0& x, const AtomsPackObj0& y, const PtensorMapObj& map){
      auto[in,out]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	direct.push_back(x.index_of(out_tensor),y.index_of(in_tensor));
      }

      return GatherMapProgram({x.nrows(),1},{y.nrows(),1},new cnine::GatherMapB(direct));
    };
  

    // 0 <- 1
    GatherMapProgram mmap(const AtomsPackObj0& x, const AtomsPackObj1& y, const PtensorMapObj& map){
      auto[in,out]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	int k=in.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(x.index_of(out_tensor),y.index_of(in_tensor,in(m,j)));
      }

      return GatherMapProgram({x.nrows(),1},{y.nrows(),1},new cnine::GatherMapB(direct));
    }


    // 0 <- 2
    GatherMapProgram mmap(const AtomsPackObj0& x, const AtomsPackObj2& y, const PtensorMapObj& map){
      auto[in,out]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in.size(); m++){
	int in_tensor=in.head(m);
	int out_tensor=out.head(m);
	vector<int> inv=in(m);
	vector<int> outv=out(m);
	for(int i0=0; i0<inv.size(); i0++)
	  direct.push_back(2*x.index_of(out_tensor)+1,y.index_of(in_tensor,inv[i0],inv[i0]));
	for(int i0=0; i0<inv.size(); i0++)
	  for(int i1=0; i1<inv.size(); i1++)
	    direct.push_back(2*x.index_of(out_tensor),y.index_of(in_tensor,inv[i0],inv[i1]));
      }

      return GatherMapProgram({x.nrows(),1},{y.nrows(),1},new cnine::GatherMapB(direct,2));
    }


  private: // ---- First order ------------------------------------------------------------------------------------


    // 1 <- 0
    GatherMapProgram mmap(const AtomsPackObj1& x, const AtomsPackObj0& y, const PtensorMapObj& map){
      auto[in_lists,out_lists]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=out_lists.size_of(m);
	for(int j=0; j<k; j++)
	  direct.push_back(x.index_of(out_tensor,out_lists(m,j)),y.index_of(in_tensor));
      }
      
      return GatherMapProgram({x.nrows(),1},{y.nrows(),1},new cnine::GatherMapB(direct));
    }
  

    // 1 <- 1
    GatherMapProgram mmap(const AtomsPackObj1& x, const AtomsPackObj1& y, const PtensorMapObj& map){
      auto[in_lists,out_lists]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	int k=in_lists.size_of(m);
	for(int j=0; j<k; j++){
	  direct.push_back(2*x.index_of(out_tensor,out_lists(m,j))+1,y.index_of(in_tensor,in_lists(m,j)));
	}
      }

      GatherMapProgram R({x.nrows(),1},{y.nrows(),1});
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(reduce0(y,in_lists),2,0);
      R.add_map(broadcast0(x,out_lists,2),1,2);
      R.add_map(new cnine::GatherMapB(direct,2));
      return R;
    }


    // 1 <- 2
    GatherMapProgram mmap(const AtomsPackObj1& x, const AtomsPackObj2& y, const PtensorMapObj& map){
      auto[in_lists,out_lists]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<in.size(); i0++)
	  direct.push_back(5*x.index_of(out_tensor,out[i0])+4,y.index_of(in_tensor,in[i0],in[i0]));
	for(int i0=0; i0<in.size(); i0++){
	  for(int i1=0; i1<in.size(); i1++){
	    direct.push_back(5*x.index_of(out_tensor,out[i0])+3,y.index_of(in_tensor,in[i0],in[i1]));
	    direct.push_back(5*x.index_of(out_tensor,out[i0])+2,y.index_of(in_tensor,in[i1],in[i0]));
	  }
	}
      }
	
      GatherMapProgram R({x.nrows(),1},{y.nrows(),1});
      R.add_var(Gdims(in_lists.size(),2));
      R.add_map(reduce0(y,in_lists),2,0);
      R.add_map(broadcast0(x,out_lists,5,0,2),1,2);
      R.add_map(new cnine::GatherMapB(direct,5));
      return R;
    }


    cnine::GatherMapB reduce0(const AtomsPackObj1& x, const cnine::hlists<int>& in_lists, const int in_columns=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	in_lists.for_each_of(m,[&](const int i){
	    R.push_back(m,in_columns*x.index_of(in_tensor,i)+coffs);});
      }
      return cnine::GatherMapB(R,1,in_columns);
    }

    cnine::GatherMapB broadcast0(const AtomsPackObj1& x, const cnine::hlists<int>& out_lists, const int stride=1, const int coffs=0, const int out_cols_n=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(stride>=1);
      PTENS_ASSRT(coffs<=stride-1);
      for(int m=0; m<out_lists.size(); m++){
	int out_tensor=out_lists.head(m);
	out_lists.for_each_of(m,[&](const int i){
	    R.push_back(stride*x.index_of(out_tensor,i)+coffs,m);});
      }
      return cnine::GatherMapB(R,stride,1,out_cols_n);
    }


  private: // ---- Second order ------------------------------------------------------------------------------------


    // 2 <- 0 
    GatherMapProgram mmap(const AtomsPackObj2& x, const AtomsPackObj0& y, const PtensorMapObj& map){
      auto[in_lists,out_lists]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<out.size(); i0++)
	  direct.push_back(2*x.index_of(out_tensor,out[i0],out[0])+1,y.index_of(in_tensor));
	for(int i0=0; i0<out.size(); i0++)
	  for(int i1=0; i1<out.size(); i1++)
	    direct.push_back(2*x.index_of(out_tensor,out[i0],out[i1]),y.index_of(in_tensor));
      }
      return GatherMapProgram({x.nrows(),1},{y.nrows(),1},new cnine::GatherMapB(direct,2));
    }
  
      
    // 2 <- 1
    GatherMapProgram mmap(const AtomsPackObj2& x, const AtomsPackObj1& y, const PtensorMapObj& map){
      auto[in_lists,out_lists]=map.ipacks();

      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<out.size(); i0++){
	  int source=y.index_of(in_tensor,in[i0]);
	  direct.push_back(5*x.index_of(out_tensor,out[i0],out[i0])+4,source);
	  for(int i1=0; i1<out.size(); i1++){
	    direct.push_back(5*x.index_of(out_tensor,out[i0],out[i1])+3,source);
	    direct.push_back(5*x.index_of(out_tensor,out[i1],out[i0])+2,source);
	  }
	}
      }

      GatherMapProgram R({x.nrows(),1},{y.nrows(),1});
      R.add_var(Gdims(in_lists.size(),1));
      R.add_map(reduce0(y,in_lists),2,0);
      R.add_map(broadcast0(x,out_lists,5),1,2);
      R.add_map(new cnine::GatherMapB(direct,5));
      return R;
    }


    // 2 <- 2
    GatherMapProgram mmap(const AtomsPackObj2& x, const AtomsPackObj2& y, const PtensorMapObj& map){
      auto[in_lists,out_lists]=map.ipacks();
	
      cnine::map_of_lists<int,int> direct;
      for(int m=0; m<in_lists.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	vector<int> in=in_lists(m);
	vector<int> out=out_lists(m);
	for(int i0=0; i0<in.size(); i0++){
	  for(int i1=0; i1<in.size(); i1++){
	    direct.push_back(15*x.index_of(out_tensor,out[i0],out[i1])+13,y.index_of(in_tensor,in[i0],in[i1]));
	    direct.push_back(15*x.index_of(out_tensor,out[i0],out[i1])+14,y.index_of(in_tensor,in[i1],in[i0]));
	  }
	}
      }

      GatherMapProgram R({x.nrows(),1},{y.nrows(),1});
      R.add_var(Gdims(in_lists.size(),2));
      R.add_map(reduce0(y,in_lists),2,0);
      R.add_map(broadcast0(x,out_lists,15,0,2),1,2);

      R.add_var(Gdims(in_lists.get_tail()-in_lists.size(),3));
      R.add_map(reduce1(y,in_lists),3,0);
      R.add_map(broadcast1(x,out_lists,15,4,3),1,3);

      R.add_map(new cnine::GatherMapB(direct,15));
      return R;
    }
      

    cnine::GatherMapB reduce0(const AtomsPackObj2& x, const cnine::hlists<int>& in_lists, const int in_columns=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(coffs<=in_columns-1);

      for(int m=0; m<in_lists.size(); m++){
	
	int in_tensor=in_lists.head(m);
	vector<int> ix=in_lists(m);
	int k=ix.size();
	
	int offs=x.offset(in_tensor);
	int n=x.size_of(in_tensor);
	
	for(int i0=0; i0<k; i0++)
	  R.push_back(2*m+1,in_columns*(offs+(n+1)*ix[i0])+coffs);
	for(int i0=0; i0<k; i0++)
	  for(int i1=0; i1<k; i1++)
	    R.push_back(2*m,in_columns*(offs+ix[i0]*n+ix[i1])+coffs);

      }
      return cnine::GatherMapB(R,2,in_columns);
    }


    cnine::GatherMapB reduce1(const AtomsPackObj2& x, const cnine::hlists<int>& in_lists, const int in_columns=1, const int coffs=0) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(coffs<=in_columns-1);

      for(int m=0; m<in_lists.size(); m++){
	
	int in_tensor=in_lists.head(m);
	vector<int> ix=in_lists(m);
	int k=ix.size();
	
	int offs=/*atoms->*/x.offset(in_tensor);
	int n=x.size_of(in_tensor);

	int out_offs=in_lists.offset(m)-m; 
	
	for(int i0=0; i0<k; i0++){
	  int target=3*(out_offs+i0);
	  R.push_back(target+2,in_columns*(offs+(n+1)*ix[i0])+coffs);
	  for(int i1=0; i1<k; i1++){
	    R.push_back(target+1,in_columns*(offs+ix[i0]*n+ix[i1])+coffs);
	    R.push_back(target,in_columns*(offs+ix[i1]*n+ix[i0])+coffs);
	  }
	}
      }
      return cnine::GatherMapB(R,3,in_columns);
    }


    cnine::GatherMapB broadcast0(const AtomsPackObj2& x, const cnine::hlists<int>& out_lists, const int ncols=2, const int coffs=0, 
      const int cstride=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(ncols>=2);
      PTENS_ASSRT(coffs<=ncols-2);

      for(int m=0; m<out_lists.size(); m++){
	
	int out_tensor=out_lists.head(m);
	vector<int> ix=out_lists(m);
	int k=ix.size();
	
	int offs=x.offset(out_tensor);
	int n=x.size_of(out_tensor);
	
	for(int i0=0; i0<k; i0++)
	  R.push_back(ncols*(offs+(n+1)*ix[i0])+coffs+cstride,m);
	for(int i0=0; i0<k; i0++)
	  for(int i1=0; i1<k; i1++)
	    R.push_back(ncols*(offs+ix[i0]*n+ix[i1])+coffs,m);

      }
      return cnine::GatherMapB(R,ncols,1,cstride);
    }


    cnine::GatherMapB broadcast1(const AtomsPackObj2& x, const cnine::hlists<int>& out_lists, const int ncols=3, const int coffs=0, const int cstride=1) const{
      cnine::map_of_lists<int,int> R;
      PTENS_ASSRT(ncols>=3);
      PTENS_ASSRT(coffs<=ncols-3);

      for(int m=0; m<out_lists.size(); m++){
	
	int out_tensor=out_lists.head(m);
	vector<int> ix=out_lists(m);
	int k=ix.size();
	
	int offs=x.offset(out_tensor);
	int n=x.size_of(out_tensor);
	
	int in_offs=out_lists.offset(m)-m;

	for(int i0=0; i0<k; i0++){
	  int source=in_offs+i0;
	  R.push_back(ncols*(offs+(n+1)*ix[i0])+coffs+2*cstride,source);
	  for(int i1=0; i1<k; i1++){
	    R.push_back(ncols*(offs+ix[i0]*n+ix[i1])+coffs+cstride,source);
	    R.push_back(ncols*(offs+ix[i1]*n+ix[i0])+coffs,source);
	  }
	}
      }
      return cnine::GatherMapB(R,ncols,1,cstride);
    }


  };

}

#endif 
