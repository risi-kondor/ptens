// ---- Reductions ---------------------------------------------------------------------------------


TENSOR reduce0(const int offs=0, int nc=0) const{
  TimedFn T("Ptensors1","reduce0",*this);
  int N=size();
  int dev=get_dev();
  if(nc==0) nc=get_nc()-offs;

  if(atoms.constk()>0){
    return view3(offs,nc).sum(1);
  }

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

  if(atoms.constk()>0){
    R+=view3(offs,nc).sum(1);
    return;
  }

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



    
