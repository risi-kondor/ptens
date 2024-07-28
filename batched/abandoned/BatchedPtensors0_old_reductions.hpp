    /*
    BatchedPtensors0 reduce0(const BatchedAtomsPack<0>& _atoms, const BatchedAindexPack& list) const{
      PTENS_DEPRECATED();
      TimedFn T("BatchedPtensors0","reduce0",*this,list,list.size()*get_nc());
      cnine::using_vram_manager vv(ptens_global::vram_manager);
      BatchedPtensors0 R(_atoms,get_nc(),0,get_dev());
      for(int i=0; i<size(); i++){
	R.view_of(i)+=view_of(i).reduce0(_atoms[i],list[i]);
      }
      return R;
    }
    */

    /*
    void broadcast0(const BatchedPtensors0& x, const BatchedAindexPack& list, const int offs=0){
      PTENS_DEPRECATED();
      for(int i=0; i<size(); i++)
	view_of(i).broadcast0(x.view_of(i),list[i],offs);
    }
    */

