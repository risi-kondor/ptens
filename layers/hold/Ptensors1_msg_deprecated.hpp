// ---- Deprecated Indexed Reductions -------------------------------------------------------------------------


Ptensors0<TYPE> reduce0(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors1","reduce0",*this,list,list.count1*nc);
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors0<TYPE> R(_atoms,nc,0,get_dev());
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      view_of(list.tens(i),list.ix(i)).sum0_into(R.view_of(i));
    }
  }
  return R;
}


Ptensors1<TYPE> reduce1(const AtomsPack& _atoms, const AindexPack& list, const int offs=0, int nc=0) const{
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors1","reduce1",*this,list,list.count1*nc);
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors1<TYPE> R(_atoms,nc,0,get_dev());
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      R.view_of(i)+=view_of(list.tens(i),list.ix(i));
    }
  }
  return R;
}

void add_reduce0_to(const Ptensors0<TYPE>& R, const AindexPack& list, const int offs=0) const{
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors1","reduce0",*this,list,list.count1*nc);
  int nc=R.get_nc();
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      view_of(list.tens(i),list.ix(i)).sum0_into(R.view_of(i));
    }
  }
}

void add_reduce1_to(const Ptensors1& R, const AindexPack& list, const int offs=0) const{
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors1","reduce1",*this,list,list.count1*nc);
  int nc=R.get_nc();
  if(dev==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      R.view_of(i)+=view_of(list.tens(i),list.ix(i));
    }
  }
}


// ---- Deprecated Indexed Broadcasting ----------------------------------------------------------------------


void broadcast0(const Ptensors0<TYPE>& x, const AindexPack& list, const int offs=0){
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors1","brcast0",*this,x,list,list.count1*x.get_nc());
  if(dev==0){
    int N=list.size();
    const int nc=x.get_nc();
    for(int i=0; i<N; i++){
      view_of(list.tens(i),list.ix(i),offs,nc)+=repeat0(x.view_of(i),list.nix(i));
    }
  }
  GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,list,offs,stream)));
}

void broadcast1(const Ptensors1<TYPE>& x, const AindexPack& list, const int offs=0){
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors1","brcast1",*this,x,list,list.count1*x.get_nc());
  if(dev==0){
    int N=list.size();
    const int nc=x.get_nc();
    for(int i=0; i<N; i++){
      if(x.size_of(i)==0) continue;
      view_of(list.tens(i),list.ix(i),offs,nc)+=x.view_of(i);
    }
  }
}
