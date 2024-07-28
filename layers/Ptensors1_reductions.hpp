// ---- Reductions ---------------------------------------------------------------------------------


TENSOR reduce0(const int offs=0, int nc=0) const{
  TimedFn T("Ptensors1","reduce0",*this);
  int N=size();
  int dev=get_dev();
  if(nc==0) nc=get_nc()-offs;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({N,nc},0,dev);
  Rtensor2_view r=R.view2();
  for(int i=0; i<N; i++)
    view_of(i,offs,nc).sum0_into(r.slice0(i));
  return R;
}

TENSOR reduce1() const{
  return *this;
}

void add_reduce0_to(const TENSOR& R, const int offs=0) const{
  PTENS_ASSRT(R.ndims()==2);
  PTENS_ASSRT(R.dim(0)==size());
  PTENS_CPUONLY();
  int N=size();
  int nc=R.dim(1);
  Rtensor2_view r=R.view2();
  for(int i=0; i<N; i++)
    view_of(i,offs,nc).sum0_into(r.slice0(i));
}


public: // ---- Indexed Reductions -------------------------------------------------------------------------


TENSOR reduce0(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors1","reduce0",*this,map,map.count1*nc);
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,get_dev());
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){
      x.sum0_into(r);
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,map,offs,nc,stream)));
  return R;
}


TENSOR reduce1(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors1","reduce1",*this,map,map.count1*nc);
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,get_dev());
  if(dev==0) zip1(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,map,offs,nc,stream)));
  return R;
}


void add_reduce0(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){x.sum0_into(r);},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,map,offs,nc,stream)));
}


void add_reduce1(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip1(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,map,offs,nc,stream)));
}


// ---- Deprecated Indexed Reductions -------------------------------------------------------------------------


/*
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
*/

    
/*
public: // ---- Indexed Reductions -------------------------------------------------------------------------

TENSOR reduce0(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors1","reduce0",*this,map,map.count1*nc);
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,get_dev());
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){
      x.sum0_into(r);
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,map,offs,nc,stream)));
  return R;
}


TENSOR reduce1(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors1","reduce1",*this,map,map.count1*nc);
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,get_dev());
  if(dev==0) zip1(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,map,offs,nc,stream)));
  return R;
}


void add_reduce0(TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){x.sum0_into(r);},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce0_cu(R,*this,map,offs,nc,stream)));
}


void add_reduce1(TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip1(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors1_reduce1_cu(R,*this,map,offs,nc,stream)));
}
*/

