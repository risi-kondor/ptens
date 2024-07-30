public: // ---- Reductions ---------------------------------------------------------------------------------


BASE reduce0() const{
  TimedFn T("Ptensors2","reduce0",*this);
  int N=size();
  int nc=get_nc();
  int dev=get_dev();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  BASE R({N,2*nc},0,dev);

  if(atoms.constk()>0){
    R.cols(0,nc)+=view4().sum(1).sum(1);
    R.cols(nc,nc)+=view4().diag({1,2}).sum(1);
    return R;
  }

  PTENS_CPUONLY();
  Rtensor2_view r=R.view2();
  if(dev==0){
    Rtensor2_view r0=R.block(0,0,N,nc);
    Rtensor2_view r1=R.block(0,nc,N,nc);
    for(int i=0; i<N; i++){
      view3_of(i).sum01_into(r0.slice0(i));
      view3_of(i).diag01().sum0_into(r1.slice0(i));
    }
  }
  return R;
}


BASE reduce0_shrink(const int offs, const int nc) const{
  TimedFn T("Ptensors2","reduce0_shrink",*this);
  int N=size();
  int dev=get_dev();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  BASE R({N,nc},0,dev);
  Rtensor2_view r=R.view2();

  if(atoms.constk()>0){
    R+=view4().cols(offs,nc).sum(1).sum(1);
    R+=view4().cols(offs+nc,nc).diag({1,2}).sum(1);
    return R;
  }

  PTENS_CPUONLY();
  if(dev==0){
    for(int i=0; i<N; i++){
      view3_of(i,offs,nc).sum01_into(r.slice0(i));
      view3_of(i,offs+nc,nc).diag01().sum0_into(r.slice0(i));
    }
  }
  return R;
}


BASE reduce1() const{
  TimedFn T("Ptensors2","reduce1",*this);
  int N=size();
  int nc=get_nc();
  int dev=get_dev();
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  BASE R({atoms.nrows1(),3*nc},0,dev);
  Rtensor2_view r=R.view2();

  if(atoms.constk()>0){
    R.cols(0,nc)+=view4().sum(1).fuse({0,1});
    R.cols(nc,nc)+=view4().sum(2).fuse({0,1});
    R.cols(2*nc,nc)+=view4().diag({1,2}).fuse({0,1});
    return R;
  }

  PTENS_CPUONLY();
  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset1(i);
      int n=size_of(i);
      view3_of(i).sum0_into(r.block(roffs,0,n,nc));
      view3_of(i).sum1_into(r.block(roffs,nc,n,nc));
      r.block(roffs,2*nc,n,nc)+=view3_of(i).diag01();
    }
  }
  return R;
}

    
BASE reduce1_shrink(const int offs, const int nc) const{
  TimedFn T("Ptensors2","reduce1_shrink",*this);
  int N=size();
  int dev=get_dev();

  cnine::using_vram_manager vv(ptens_global::vram_manager);
  BASE R({atoms.nrows1(),nc},0,dev);
  Rtensor2_view r=R.view2();

  if(atoms.constk()>0){
    R+=view4().cols(offs,nc).sum(1).fuse({0,1});
    R+=view4().cols(offs+nc,nc).sum(1).fuse({0,1});
    R+=view4().cols(offs+2*nc,nc).diag({1,2}).fuse({0,1});
    return R;
  }

  PTENS_CPUONLY();
  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset1(i);
      int n=size_of(i);
      view3_of(i,offs,nc).sum0_into(r.block(roffs,0,n,nc));
      view3_of(i,offs+nc,nc).sum1_into(r.block(roffs,0,n,nc));
      r.block(roffs,0,n,nc)+=view3_of(i,offs+2*nc,nc).diag01();
    }
  }
  return R;
}


BASE reduce2_shrink(const int offs, const int nc) const{
  TimedFn T("Ptensors2","reduce2_shrink",*this);
  int N=size();
  int dev=get_dev();
      
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  BASE R({dim(0),nc},0,dev);
  Rtensor2_view r=R.view2();

  if(atoms.constk()>0){
    R+=view4().cols(offs,nc);
    R+=view4().cols(offs+nc,nc).transp(1,2);
    return R;
  }

  PTENS_CPUONLY();
  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset(i);
      int n=size_of(i);
      r.block(roffs,0,n*n,nc)+=view3_of(i,offs,nc).fuse01();
      r.block(roffs,0,n*n,nc)+=view3_of(i,offs+nc,nc).transp01().fuse01();
    }
  }
  return R;
}


public: // ---- Cumulative Reductions ----------------------------------------------------------------------


void add_reduce0_to(const BASE& R) const{
  TimedFn T("Ptensors2","reduce0",*this);
  PTENS_ASSRT(R.ndims()==2);
  PTENS_ASSRT(R.dim(0)==size());
  PTENS_ASSRT(R.dim(1)==2*nc);
  PTENS_CPUONLY();
  int N=size();
  int dev=get_dev();
  //Rtensor2_view r=R.view2();
  if(dev==0){
    Rtensor2_view r0=R.block(0,0,N,nc);
    Rtensor2_view r1=R.block(0,nc,N,nc);
    for(int i=0; i<N; i++){
      view3_of(i).sum01_into(r0.slice0(i));
      view3_of(i).diag01().sum0_into(r1.slice0(i));
    }
  }
}


void add_reduce0_shrink_to(const BASE& R, const int offs) const{
  TimedFn T("Ptensors2","reduce0_shrink",*this);
  PTENS_ASSRT(R.ndims()==2);
  PTENS_ASSRT(R.dim(0)==size());
  PTENS_CPUONLY();
  int N=size();
  int nc=R.dim(1);
  int dev=get_dev();
  Rtensor2_view r=R.view2();
  if(dev==0){
    for(int i=0; i<N; i++){
      view3_of(i,offs,nc).sum01_into(r.slice0(i));
      view3_of(i,offs+nc,nc).diag01().sum0_into(r.slice0(i));
    }
  }
}


void add_reduce1_to(const BASE& R) const{
  TimedFn T("Ptensors2","reduce1",*this);
  PTENS_ASSRT(R.ndims()==2);
  PTENS_ASSRT(R.dim(0)==atoms.nrows1());
  PTENS_ASSRT(R.dim(1)==3*nc);
  PTENS_CPUONLY();
  int N=size();
  int dev=get_dev();
  Rtensor2_view r=R.view2();
  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset1(i);
      int n=size_of(i);
      view3_of(i).sum0_into(r.block(roffs,0,n,nc));
      view3_of(i).sum1_into(r.block(roffs,nc,n,nc));
      r.block(roffs,2*nc,n,nc)+=view3_of(i).diag01();
    }
  }
}

    
void add_reduce1_shrink_to(const BASE& R, const int offs) const{
  TimedFn T("Ptensors2","reduce1_shrink",*this);
  PTENS_ASSRT(R.ndims()==2);
  PTENS_ASSRT(R.dim(0)==atoms.nrows1());
  PTENS_CPUONLY();
  int N=size();
  int nc=R.dim(1);
  int dev=get_dev();
  Rtensor2_view r=R.view2();
  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset1(i);
      int n=size_of(i);
      view3_of(i,offs,nc).sum0_into(r.block(roffs,0,n,nc));
      view3_of(i,offs+nc,nc).sum1_into(r.block(roffs,0,n,nc));
      r.block(roffs,0,n,nc)+=view3_of(i,offs+2*nc,nc).diag01();
    }
  }
}


void add_reduce2_shrink_to(const BASE& R, const int offs) const{
  TimedFn T("Ptensors2","reduce2_shrink",*this);
  PTENS_ASSRT(R.ndims()==2);
  PTENS_ASSRT(R.dim(0)==dim(0));
  PTENS_CPUONLY();
  int N=size();
  int nc=R.dim(1);
  int dev=get_dev();
  Rtensor2_view r=R.view2();
  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset(i);
      int n=size_of(i);
      r.block(roffs,0,n*n,nc)+=view3_of(i,offs,nc).fuse01();
      r.block(roffs,0,n*n,nc)+=view3_of(i,offs+nc,nc).transp01().fuse01();
    }
  }
}


public: // ---- Indexed reductions -------------------------------------------------------------------------


TENSOR reduce0(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce0",*this,map,(map.count2+map.count1)*get_nc());
  if (nc==0) nc=get_nc()-offs;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,2*nc},0,dev);
  if(dev==0) zip0(map,R,[nc](auto& r, auto& x, int k){
      x.sum01_into(r.block(0,nc));
      x.diag01().sum0_into(r.block(nc,nc));
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce0_cu(R,*this,map,offs,nc,stream)));
  return R;
}

void add_reduce0(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1)/2;
  if(dev==0) zip0(map,R,[nc](auto& r, auto& x, int k){
      x.sum01_into(r.block(0,nc));
      x.diag01().sum0_into(r.block(nc,nc));
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce0_cu(R,*this,map,offs,nc,stream)));
}

TENSOR reduce0_shrink(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce0",*this,map,(map.count2+map.count1)*nc);
  if(nc==0) nc=(get_nc()-offs)/2;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,dev);
  if(dev==0) zip0(map,R,[nc](auto& r, auto& x, int k){
      x.cols(0,nc).sum01_into(r);
      x.cols(nc,nc).diag01().sum0_into(r);
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce0_shrink_cu(R,*this,map,offs,nc,stream)));
  return R;
}

void add_reduce0_shrink(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip0(map,R,[nc](auto& r, auto& x, int k){
      x.cols(0,nc).sum01_into(r);
      x.cols(nc,nc).diag01().sum0_into(r);
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce0_shrink_cu(R,*this,map,offs,nc,stream)));
}

TENSOR reduce1(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce1",*this,map,(map.count1+2*map.count2)*get_nc());
  if (nc==0) nc=get_nc()-offs;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,3*nc},0,dev);
  if(dev==0) zip1(map,R,[nc](auto& r, auto& x, int k){
      x.sum0_into(r.cols(0,nc));
      x.sum1_into(r.cols(nc,nc));
      r.cols(2*nc,nc)+=x.diag01();
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce1_cu(R,*this,map,offs,nc,stream)));
  return R;
}

void add_reduce1(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1)/3;
  if(dev==0) zip1(map,R,[nc](auto& r, auto& x, int k){
      x.sum0_into(r.cols(0,nc));
      x.sum1_into(r.cols(nc,nc));
      r.cols(2*nc,nc)+=x.diag01();
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce1_cu(R,*this,map,offs,nc,stream)));
}

TENSOR reduce1_shrink(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce1",*this,map,(map.count1+2*map.count2)*nc);
  if(nc==0) nc=(get_nc()-offs)/3;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,dev);
  if(dev==0) zip1(map,R,[nc](auto& r, auto& x, int k){
      x.cols(0,nc).sum0_into(r);
      x.cols(nc,nc).sum1_into(r);
      r+=x.cols(2*nc,nc).diag01();
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce1_shrink_cu(R,*this,map,offs,nc,stream)));
  return R;
}


void add_reduce1_shrink(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip1(map,R,[nc](auto& r, auto& x, int k){
      x.cols(0,nc).sum0_into(r);
      x.cols(nc,nc).sum1_into(r);
      r+=x.cols(2*nc,nc).diag01();
    },offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce1_shrink_cu(R,*this,map,offs,nc,stream)));
}


TENSOR reduce2(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce2",*this,map,(map.count1+2*map.count2)*nc);
  if(nc==0) nc=(get_nc()-offs);
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,dev);
  if(dev==0) zip2(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce2_cu(R,*this,map,offs,nc,stream)));
  return R;
}


void add_reduce2(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip2(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce2_cu(R,*this,map,offs,nc,stream)));
}


TENSOR reduce2_shrink(const AindexPackB& map, const int offs=0, int nc=0) const{
  TimedFn T("Ptensors2","reduce2",*this,map,(map.count1+2*map.count2)*nc);
  if(nc==0) nc=(get_nc()-offs)/2;
  cnine::using_vram_manager vv(ptens_global::vram_manager);
  TENSOR R({map.nrows,nc},0,dev);
  if(dev==0) zip2(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  if(dev==0) zip2(map,R,[](auto& r, auto& x, int k){r+=x.transp();},offs+nc,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce2_shrink_cu(R,*this,map,offs,nc,stream)));
  return R;
}


void add_reduce2_shrink(const TENSOR& R, const AindexPackB& map, const int offs=0) const{
  int nc=R.dim(1);
  if(dev==0) zip2(map,R,[](auto& r, auto& x, int k){r+=x;},offs,nc);
  if(dev==0) zip2(map,R,[](auto& r, auto& x, int k){r+=x.transp();},offs+nc,nc);
  GPUCODE(CUDA_STREAM(Ptensors2_reduce2_shrink_cu(R,*this,map,offs,nc,stream)));
}

