

void broadcast0(const TENSOR& X, const int offs=0){
  int nc=X.dim(1);
  BASE::view2().cols(offs,nc)+=X.view2();
}

void broadcast0(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors0","broadcast0",*this,x,map,map.size()*get_nc());
  if(dev==0) zip0(map,x,[](auto& r, auto& x, int k){x+=r;},offs);
  GPUCODE(CUDA_STREAM(Ptensors0_broadcast0_cu(*this,x,map,offs,stream)));
}

