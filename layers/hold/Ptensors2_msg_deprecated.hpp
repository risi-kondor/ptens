public: // ---- Deprecated Indexed reductions -------------------------------------------------------------------------

Ptensors0<TYPE> reduce0(const AtomsPack& _atoms, const AindexPack& list, const int offs=0) const{
  TimedFn T("Ptensors2","reduce0",*this,list,(list.count2+list.count1)*get_nc());
  int nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors0<TYPE> R(_atoms,2*nc,0,dev);
  add_reduce0_to(R,list);
  return R;
}

void add_reduce0_to(const Ptensors0<TYPE>& R, const AindexPack& list, const int offs=0) const{
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      view_of(list.tens(i),list.ix(i)).sum01_into(R.view_of(i).block(0,nc));
      view_of(list.tens(i),list.ix(i)).diag01().sum0_into(R.view_of(i).block(nc,nc));
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_reduce0_cu(R,*this,list,0,nc,stream)));
}


Ptensors0<TYPE> reduce0_shrink(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce0",*this,list,(list.count2+list.count1)*nc);
  if(nc==0) nc=get_nc()/2;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors0<TYPE> R(_atoms,nc,0,dev);
  add_reduce0_shrink_to(R,list,offs);
  return R;
}

void add_reduce0_shrink_to(const Ptensors0<TYPE>& R, const AindexPack& list, const int offs=0) const{
  int nc=R.get_nc();
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      view_of(list.tens(i),list.ix(i),offs,nc).sum01_into(R.view_of(i));
      view_of(list.tens(i),list.ix(i),offs+nc,nc).diag01().sum0_into(R.view_of(i));
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_reduce0_cu(R,*this,list,0,nc,stream)));
}


Ptensors1<TYPE> reduce1(const AtomsPack& _atoms, const AindexPack& list, const int offs=0) const{
  TimedFn T("Ptensors2","reduce1",*this,list,(list.count1+2*list.count2)*get_nc());
  int nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors1<TYPE> R(_atoms,3*nc,0,dev);
  add_reduce1_to(R,list);
  return R;
}

void add_reduce1_to(const Ptensors1<TYPE>& R, const AindexPack& list, const int offs=0) const{
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      view_of(list.tens(i),list.ix(i)).sum0_into(R.view_of(i).block(0,0,-1,nc));
      view_of(list.tens(i),list.ix(i)).sum1_into(R.view_of(i).block(0,nc,-1,nc));
      R.view_of(i).block(0,2*nc,-1,nc)+=view_of(list.tens(i),list.ix(i)).diag01(); // is this a problem?
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_reduce1_cu(R,*this,list,0,nc,stream)));
}


Ptensors1<TYPE> reduce1_shrink(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce1",*this,list,(list.count1+2*list.count2)*nc);
  if(nc==0) nc=get_nc()/3;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors1<TYPE> R(_atoms,nc,0,dev);
  add_reduce1_shrink_to(R,list,offs,nc);
  return R;
}

void add_reduce1_shrink_to(const Ptensors1<TYPE>& R, const AindexPack& list, const int offs=0, int nc=0) const{
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      view_of(list.tens(i),list.ix(i),offs,nc).sum0_into(R.view_of(i));
      view_of(list.tens(i),list.ix(i),offs+nc,nc).sum1_into(R.view_of(i));
      R.view_of(i)+=view_of(list.tens(i),list.ix(i),offs+2*nc,nc).diag01(); // is this a problem?
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_reduce1_cu(R,*this,list,0,nc,stream)));
}


Ptensors2<TYPE> reduce2(const AtomsPack& _atoms, const AindexPack& list, const int offs=0) const{
  TimedFn T("Ptensors2","reduce2",*this,list,(2*list.count2)*get_nc());
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors2<TYPE> R(_atoms,get_nc(),0,dev);
  add_reduce2_to(R,list);
  return R;
}

void add_reduce2_to(const Ptensors2<TYPE>& R, const AindexPack& list, const int offs=0) const{
  if(dev==0){
    int nc=get_nc();
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      R.view3_of(i,0,nc)+=view_of(list.tens(i),list.ix(i));
      R.view3_of(i,nc,nc)+=view_of(list.tens(i),list.ix(i)).transp();
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_reduce2B_cu(R,*this,list,offs,nc,stream)));
}


Ptensors2<TYPE> reduce2_shrink(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce2",*this,list,(2*list.count2)*nc);
  if(nc==0) nc=get_nc()/2;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors2<TYPE> R(_atoms,nc,0,dev);
  add_reduce2_shrink_to(R,list,offs,nc);
  return R;
}

void add_reduce2_shrink_to(const Ptensors2<TYPE>& R, const AindexPack& list, const int offs=0, int nc=0) const{
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs,nc);
      R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs+nc,nc).transp();
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_reduce2B_cu(R,*this,list,offs,nc,stream)));
}


public: // ---- Deprecated Idexed broadcasting -------------------------------------------------------------------------------


void broadcast0(const Ptensors0<TYPE>& x, const AindexPack& list, const int offs=0){
  TimedFn T("Ptensors2","broadcast0",*this,x,list,(list.count1+list.count2)*x.get_nc());
  if(dev==0){
    const int n=x.get_nc();
    int N=list.size();
    for(int i=0; i<N; i++){
      if(x.size_of(i)==0) continue; // probably redundant
      view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(repeat0(x.view_of(i),list.nix(i)),list.nix(i));
      view_of(list.tens(i),list.ix(i),offs+n,n).diag01()+=repeat0(x.view_of(i),list.nix(i));
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast0_cu(*this,x,list,offs,stream)));
}

void broadcast1(const Ptensors1<TYPE>& x, const AindexPack& list, const int offs=0){
  TimedFn T("Ptensors2","broadcast1",*this,x,list,(list.count1+2*list.count2)*x.get_nc());
  if(dev==0){
    const int n=x.get_nc();
    int N=list.size();
    for(int i=0; i<N; i++){
      if(x.size_of(i)==0) continue;
      view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view_of(i),list.nix(i));
      view_of(list.tens(i),list.ix(i),offs+n,n)+=repeat1(x.view_of(i),list.nix(i));
      view_of(list.tens(i),list.ix(i),offs+2*n,n).diag01()+=x.view_of(i);
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast1_cu(*this,x,list,offs,stream)));
}

void broadcast2(const Ptensors2<TYPE>& x, const AindexPack& list, const int offs=0){
  TimedFn T("Ptensors2","broadcast2",*this,x,list,(2*list.count2)*x.get_nc());
  if(dev==0){
    const int n=x.get_nc();
    int N=list.size();
    for(int i=0; i<N; i++){
      if(x.size_of(i)==0) continue;
      view_of(list.tens(i),list.ix(i),offs,n)+=x.view3_of(i);
      view_of(list.tens(i),list.ix(i),offs+n,n)+=x.view3_of(i).transp01();
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast2_cu(*this,x,list,offs,stream)));
}


void broadcast0_shrink(const Ptensors0<TYPE>& x, const AindexPack& list, const int offs=0){
  TimedFn T("Ptensors2","broadcast0_shrink",*this,x,list,(list.count1+list.count2)*get_nc());
  if(dev==0){
    const int n=get_nc();
    int N=list.size();
    for(int i=0; i<N; i++){
      if(x.size_of(i)==0) continue; // probably redundant
      view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(repeat0(x.view_of(i,0,n),list.nix(i)),list.nix(i));
      view_of(list.tens(i),list.ix(i),offs,n).diag01()+=repeat0(x.view_of(i,n,n),list.nix(i));
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast0_cu(*this,x,list,offs,stream)));
}

void broadcast1_shrink(const Ptensors1<TYPE>& x, const AindexPack& list, const int offs=0){
  TimedFn T("Ptensors2","broadcast1_shrink",*this,x,list,(list.count1+2*list.count2)*x.get_nc());
  if(dev==0){
    const int n=get_nc();
    int N=list.size();
    for(int i=0; i<N; i++){
      if(x.size_of(i)==0) continue;
      view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view_of(i,0,n),list.nix(i));
      view_of(list.tens(i),list.ix(i),offs,n)+=repeat1(x.view_of(i,n,n),list.nix(i));
      view_of(list.tens(i),list.ix(i),offs,n).diag01()+=x.view_of(i,2*n,n);
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast1_cu(*this,x,list,offs,stream)));
}
*/

/*
  void broadcast2_shrink(const Ptensors2<TYPE>& x, const AindexPack& list, const int offs=0){
  TimedFn T("Ptensors2","broadcast2_shrink",*this,x,list,(2*list.count2)*x.get_nc());
  if(dev==0){
  const int n=get_nc();
  int N=list.size();
  for(int i=0; i<N; i++){
  if(x.size_of(i)==0) continue;
  view_of(list.tens(i),list.ix(i),offs,n)+=x.view3_of(i);
  view_of(list.tens(i),list.ix(i),offs+n,n)+=x.view3_of(i).transp01();
  }
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast2_cu(*this,x,list,offs,stream)));
  }
