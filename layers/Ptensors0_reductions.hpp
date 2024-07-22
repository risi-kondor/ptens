

TENSOR reduce0() const{
  return *this;
}

Ptensors0<TYPE> reduce0(const AtomsPack& _atoms, const AindexPack& list) const{
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors0","reduce0",*this,list,list.size()*get_nc());
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  Ptensors0 R(_atoms,get_nc(),0,get_dev());
  if(get_dev()==0){
    int N=list.size();
    for(int i=0; i<N; i++){
      if(list.nix(i)==0) continue;
      R.view_of(i)=view_of(list.tix(i));
    }
  }
  return R;
}

TENSOR reduce0(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors0","reduce0",*this,map,map.size()*get_nc());
  if(nc==0) nc=get_nc();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,get_dev());
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,map,offs,stream)));
  return R;
}

void add_reduce0(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  TimedFn T("Ptensors0","reduce0",*this,map,map.size()*get_nc());
  PTENS_ASSRT(R.get_dev()==dev);
  int nc=R.dim(1);
  if(dev==0) zip0(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors0_reduce0_cu(R,*this,map,offs,stream)));
}


