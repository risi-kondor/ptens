pybind11::class_<ptens::GgraphCache>(m,"ggraph_cache")

  .def_static("size",[](){return ptens_global::graph_cache.size();})

  .def_static("cache",[](const int i, const Ggraph& x){ptens_global::graph_cache.cache(i,x.obj);})

  .def_static("graph",[](const int i){return Ggraph(ptens_global::graph_cache(i));})

  .def_static("str",[](){return ptens_global::graph_cache.str();});



