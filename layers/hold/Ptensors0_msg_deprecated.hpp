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

