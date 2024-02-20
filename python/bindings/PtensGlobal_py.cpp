
m.def("managed_gpu_memory",[](const int s){
    ptens::ptens_session.managed_gmem=new SimpleMemoryManager(Mbytes(s),1);});

m.def("clear_managed_gpu_memory",[](){
    if(ptens::ptens_session.managed_gmem)
      ptens::ptens_session.managed_gmem->clear();});


