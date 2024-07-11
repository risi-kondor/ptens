pybind11::class_<ptens::SubgraphCache>(m,"subgraph_cache")

  .def_static("size",[](){return ptens_global::subgraph_cache.size();})

  .def_static("torch",[](){
      vector<Subgraph> v;
      for(auto& p: ptens_global::subgraph_cache)
	v.push_back(Subgraph(p.obj));
      return v;})

  .def_static("str",[](){
      return ptens_global::subgraph_cache.str();});

//ostringstream oss;
//      for(auto& p: ptens_global::subgraph_cache)
//	oss<<p<<endl;
//      return oss.str();});



