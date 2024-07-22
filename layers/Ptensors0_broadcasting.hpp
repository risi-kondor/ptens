

void broadcast0(const TENSOR& X, const int offs=0){
  int nc=X.dim(1);
  BASE::view2().cols(offs,nc)+=X.view2();
}

void broadcast0(const Ptensors0& x, const AindexPack& list, const int offs=0){
  PTENS_DEPRECATED();
  PTENS_CPUONLY();
  TimedFn T("Ptensors0","broadcast0",*this,x,list,list.size()*get_nc());
  if(get_dev()==0){
    int N=list.size();
    const int n=x.get_nc();
    for(int i=0; i<N; i++)
      view_of(list.tix(i),offs,n)+=x.view_of(i);
  }
}

void broadcast0(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors0","broadcast0",*this,x,map,map.size()*get_nc());
  if(dev==0) zip0(map,x,[](auto& r, auto& x, int k){x+=r;},offs);
  GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,map,offs,stream)));
}


