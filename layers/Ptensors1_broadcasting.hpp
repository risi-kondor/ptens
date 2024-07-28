// ---- Broadcasting -------------------------------------------------------------------------------


void broadcast0(const TENSOR& X, const int offs=0){
  TimedFn T("Ptensors1","broadcast0",*this);
  int N=size();
  int nc=X.dim(1);
  PTENS_ASSRT(X.dim(0)==N);
  Rtensor2_view x=X.view2();
  for(int i=0; i<N; i++)
    view_of(i,offs,nc)+=cnine::repeat0(x.slice0(i),size_of(i));
}

void broadcast1(const TENSOR& X, const int offs=0){
  TimedFn T("Ptensors1","broadcast1",*this);
  int nc=X.dim(1);
  BASE::view2().block(0,offs,dim(0),nc)+=X.view2();
}


// ---- Indexed broadcasting -----------------------------------------------------------------------


void broadcast0(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors1","broadcast0",*this,x,map,map.count1*x.dim(1));
  if(dev==0) zip0(map,x,[](auto& r, auto& x, int k){
      x+=repeat0(r,k);},offs,x.dim(1));
  GPUCODE(CUDA_STREAM(Ptensors1_broadcast0_cu(*this,x,map,offs,stream)));
}

void broadcast1(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors1","brcast1",*this,x,map,map.count1*x.dim(1));
  if(dev==0) zip1(map,x,[](auto& r, auto& x, int k){x+=r;},offs,x.dim(1));
  GPUCODE(CUDA_STREAM(Ptensors1_broadcast1_cu(*this,x,map,offs,stream)));
}


    
// ---- Deprecated Indexed broadcasting -----------------------------------------------------------------------


/*
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
*/    

