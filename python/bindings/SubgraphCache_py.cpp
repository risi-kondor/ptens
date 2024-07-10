pybind11::class_<ptens::SubgraphCache>(m,"subgraph_cache")

  .def_static("size",[](){return ptens_global::subgraph_cache.size();})

  .def_static("str",[](){
      ostringstream oss;
      for(auto& p: ptens_global::subgraph_cache)
	oss<<p<<endl;
      return oss.str();});



