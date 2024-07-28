    BatchedPtensors0<TYPE> reduce0(const BatchedAtomsPack<0>& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      PTENS_DEPRECATED();
      TimedFn T("BatchedPtensors1","reduce0",*this,list,list.count1*nc);
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce0_to(R.view_of(i),list[i],offs);
      return R;
    }


    BatchedPtensors1<TYPE> reduce1(const BatchedAtomsPack<1>& _atoms, const BatchedAindexPack& list, const int offs=0, int nc=0) const{
      PTENS_DEPRECATED();
      TimedFn T("BatchedPtensors1","reduce1",*this,list,list.count1*nc);
      if(nc==0) nc=get_nc();
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors1<TYPE> R(_atoms,nc,0,get_dev());
      for(int i=0; i<size(); i++)
	view_of(i).add_reduce1_to(R.view_of(i),list[i],offs);
      return R;
    }


    void broadcast0(const BatchedPtensors0<TYPE>& x, const int offs=0){
      PTENS_DEPRECATED();
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),offs);
    }

    void broadcast1(const BASE& x, const int offs=0){
      BASE::view2().block(0,offs,dim(0),x.dim(1))+=x.view2();
    }

    void broadcast0(const BatchedPtensors0<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),list[i],offs);
    }

    void broadcast1(const BatchedPtensors1<TYPE>& x, const BatchedAindexPack& list, const int offs=0){
      for(int i=0; i<size(); i++)
	view_of(i).broadcast1(x.view_of(i),list[i],offs);
    }



