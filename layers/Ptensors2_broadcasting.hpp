public: // ---- Broadcasting -------------------------------------------------------------------------------


void broadcast0(const TENSOR& X, const int offs=0){
  TimedFn T("Ptensors2","broadcast0",*this);
  int N=size();
  int dev=get_dev();
  int nc=X.dim(1);
  PTENS_ASSRT(X.dim(0)==N);
  Rtensor2_view x=X.view2();
      
  if(atoms.constk()>0){
    view4(offs,nc).fuse({1,2}).add_broadcast(1,X);
    view4(offs+nc,nc).diag({1,2}).add_broadcast(1,X);
    return;
  }

  if(dev==0){
    for(int i=0; i<N; i++){
      int n=size_of(i);
      view3_of(i,offs,nc)+=repeat0(repeat0(x.slice0(i),n),n);
      view3_of(i,offs+nc,nc).diag01()+=repeat0(x.slice0(i),n);
    }
  }
}

void broadcast0_shrink(const TENSOR& X, int offs=0){
  TimedFn T("Ptensors2","broadcast0_shrink",*this);
  int N=size();
  int dev=get_dev();
  int nc=dim(1);
  PTENS_ASSRT(offs==0);
  PTENS_ASSRT(X.dim(0)==N);
  PTENS_ASSRT(X.dim(1)==2*nc);
  Rtensor2_view x=X.view2();
  Rtensor2_view x0=x.block(0,0,N,nc);
  Rtensor2_view x1=x.block(0,nc,N,nc);
      
  if(atoms.constk()>0){
    view4(offs,nc).fuse({1,2}).add_broadcast(1,X.cols(0,nc));
    view4(offs,nc).diag({1,2}).add_broadcast(1,X.cols(nc,nc));
    return;
  }

  if(dev==0){
    for(int i=0; i<N; i++){
      int n=size_of(i);
      view3_of(i)+=repeat0(repeat0(x0.slice0(i),n),n);
      view3_of(i).diag01()+=repeat0(x1.slice0(i),n);
    }
  }
}
    

void broadcast1(const TENSOR& X, const int offs=0){
  TimedFn T("Ptensors2","broadcast1",*this);
  int N=size();
  int dev=get_dev();
  int nc=X.dim(1);
  Rtensor2_view x=X.view2();

  if(atoms.constk()>0){
    int k=atoms.constk();
    view4(offs,nc).add_broadcast(1,X.split(0,k));
    view4(offs+nc,nc).add_broadcast(2,X.split(0,k));
    view4(offs+2*nc,nc).diag({1,2})+=X.split(0,k);
    return;
  }

  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset1(i);
      int n=size_of(i);
      view3_of(i,offs,nc)+=repeat0(x.block(roffs,0,n,nc),n);
      view3_of(i,offs+nc,nc)+=repeat1(x.block(roffs,0,n,nc),n);
      view3_of(i,offs+2*nc,nc).diag01()+=x.block(roffs,0,n,nc);
    }
  }
}


void broadcast1_shrink(const TENSOR& X, const int offs=0){
  TimedFn T("Ptensors2","broadcast1_shrink",*this);
  int N=size();
  int dev=get_dev();
  int nc=dim(1);
  PTENS_ASSRT(X.dim(1)==3*nc);
  PTENS_ASSRT(offs==0);
  Rtensor2_view x=X.view2();
  //Rtensor2_view x0=x.block(0,0,X.dim(0),nc);
  //Rtensor2_view x1=x.block(0,nc,X.dim(0),nc);
  //Rtensor2_view x2=x.block(0,2*nc,X.dim(0),nc);
      
  if(atoms.constk()>0){
    int k=atoms.constk();
    view4(offs,nc).add_broadcast(1,X.cols(0,nc).split(0,k));
    view4(offs,nc).add_broadcast(2,X.cols(nc,nc).split(0,k));
    view4(offs,nc).diag({1,2})+=X.cols(2*nc,nc).split(0,k);
    return;
  }

  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset1(i);
      int n=size_of(i);
      view3_of(i)+=repeat0(x.block(roffs,0,n,nc),n);
      view3_of(i)+=repeat1(x.block(roffs,nc,n,nc),n);
      view3_of(i).diag01()+=x.block(roffs,2*nc,n,nc);
    }
  }
}


void broadcast2(const TENSOR& X, const int offs=0){
  TimedFn T("Ptensors2","broadcast2",*this);
  int N=size();
  int dev=get_dev();
  int nc=X.dim(1);
  Rtensor2_view x=X.view2();
      
  if(atoms.constk()>0){
    int k=atoms.constk();
    cols(offs,nc)+=X;
    view4(offs+nc,nc)+=X.split(0,k).split(0,k).transp(1,2);
    return;
  }

  if(dev==0){
    for(int i=0; i<N; i++){
      int roffs=offset(i);
      int n=size_of(i);
      view3_of(i,offs,nc)+=split0(x.block(roffs,0,n*n,nc),n,n);
      view3_of(i,offs+nc,nc)+=split0(x.block(roffs,0,n*n,nc),n,n).transp01();
    }
  }
}




// ---- Indexed broadcasting -----------------------------------------------------------------------


void broadcast0(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors2","broadcast0",*this,x,map,map.count1*x.dim(1));
  int nc=x.dim(1);
  if(dev==0){
    zip0(map,x,[](auto& r, auto& x, int k){x+=repeat0(repeat0(r,k),k);},offs,nc);
    zip0(map,x,[](auto& r, auto& x, int k){x.diag01()+=repeat0(r,k);},offs+nc,nc);
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast0_cu(*this,x,map,offs,stream)));
}

void broadcast0_shrink(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors2","broadcast0",*this,x,map,map.count1*x.dim(1));
  int nc=x.dim(1)/2;
  if(dev==0){
    zip0(map,x,[nc](auto& r, auto& x, int k){x+=repeat0(repeat0(r.block(0,nc),k),k);},offs,nc);
    zip0(map,x,[nc](auto& r, auto& x, int k){x.diag01()+=repeat0(r.block(nc,nc),k);},offs,nc);
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast0_shrink_cu(*this,x,map,offs,stream)));
}

void broadcast1(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors2","broadcast1",*this,x,map,map.count1*x.dim(1));
  int nc=x.dim(1);
  if(dev==0){
    zip1(map,x,[](auto& r, auto& x, int k){x+=repeat0(r,k);},offs,nc);
    zip1(map,x,[](auto& r, auto& x, int k){x+=repeat1(r,k);},offs+nc,nc);
    zip1(map,x,[](auto& r, auto& x, int k){x.diag01()+=r;},offs+2*nc,nc);
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast1_cu(*this,x,map,offs,stream)));
}

void broadcast1_shrink(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors2","broadcast1",*this,x,map,map.count1*x.dim(1));
  int nc=x.dim(1)/3;
  if(dev==0){
    zip1(map,x,[nc](auto& r, auto& x, int k){x+=repeat0(r.cols(0,nc),k);},offs,nc);
    zip1(map,x,[nc](auto& r, auto& x, int k){x+=repeat1(r.cols(nc,nc),k);},offs,nc);
    zip1(map,x,[nc](auto& r, auto& x, int k){x.diag01()+=r.cols(2*nc,nc);},offs,nc);
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast1_shrink_cu(*this,x,map,offs,stream)));
}

void broadcast2(const TENSOR& x, const AindexPackB& map, const int offs=0){
  TimedFn T("Ptensors2","broadcast2",*this,x,map,map.count1*x.dim(1));
  int nc=x.dim(1);
  if(dev==0){
    zip2(map,x,[](auto& r, auto& x, int k){x+=r;},offs,nc);
    zip2(map,x,[](auto& r, auto& x, int k){x.transp()+=r;},offs+nc,nc);
  }
  GPUCODE(CUDA_STREAM(Ptensors2_broadcast2_cu(*this,x,map,offs,stream)));
}

