    BatchedPtensors0<TYPE> reduce0(const BatchedAtomsPack<0>& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc();
      PTENS_ASSRT(offs==0);
      PTENS_ASSRT(nc==get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0<TYPE> R(_atoms,2*nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce0_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors0<TYPE> reduce0_shrink(const BatchedAtomsPack<0>& _atoms, const BatchedAindexPack& list, const int offs, int nc) const{
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0<TYPE> R(_atoms,2*nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce0_shrink_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors1<TYPE> reduce1(const BatchedAtomsPack<1>& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc();
      PTENS_ASSRT(offs==0);
      PTENS_ASSRT(nc==get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors1<TYPE> R(_atoms,3*nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce1_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors1<TYPE> reduce1_shrink(const BatchedAtomsPack<1>& _atoms, const BatchedAindexPack& list, const int offs, int nc) const{
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors1<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce1_shrink_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors2<TYPE> reduce2(const BatchedAtomsPack<2>& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      if(nc==0) nc=get_nc();
      PTENS_ASSRT(offs==0);
      PTENS_ASSRT(nc==get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors2<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	  view_of(i).add_reduce2_to(R.view_of(i),list[i],offs);
      return R;
    }

    BatchedPtensors2<TYPE> reduce2_shrink(const BatchedAtomsPack<2>& _atoms, const BatchedAindexPack& list, const int offs, int nc) const{
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors2<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce2_shrink_to(R.view_of(i),list[i],offs);
      return R;
    }


    void broadcast0(const BatchedPtensors0<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),list[i],offs);
    }

    void broadcast0_shrink(const BatchedPtensors0<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0_shrink(x.view_of(i),list[i],offs);
    }

    void broadcast1(const BatchedPtensors1<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast1(x.view_of(i),list[i],offs);
    }

    void broadcast1_shrink(const BatchedPtensors1<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast1_shrink(x.view_of(i),list[i],offs);
    }

    void broadcast2(const BatchedPtensors2<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast2(x.view_of(i),list[i],offs);
    }

