
m.def("status_str",[](){return ptens_session.status_str();});


/*
m.def("subgraph_cache_subgraphs",[](){
    vector<Subgraph> R;
    //for(auto& p:ptens_global::subgraph_cache)
    for(auto it=ptens_global::subgraph_cache.begin(); it!=ptens_global::subgraph_cache.end(); it++)
      R.push_back(Subgraph(it));
    return R;
  });
*/ 




//m.def("managed_gpu_memory",[](const int s){
//    ptens::ptens_session->managed_gmem=new SimpleMemoryManager(Mbytes(s),1);});

//m.def("clear_managed_gpu_memory",[](){
//    if(ptens::ptens_session->managed_gmem)
//      ptens::ptens_session->managed_gmem->clear();});


