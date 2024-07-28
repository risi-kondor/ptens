
TENSOR reduce0() const{
  return *this;
}

TENSOR reduce0(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors0","reduce0",*this,map,map.size()*get_nc());
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,get_dev());
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,map,offs,nc,stream)));
  return R;
}

void add_reduce0(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  TimedFn T("Ptensors0","reduce0",*this,map,map.size()*get_nc());
  PTENS_ASSRT(R.get_dev()==dev);
  int nc=R.dim(1);
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,map,offs,nc,stream)));
}


