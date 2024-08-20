pybind11::class_<ptens::CompressedSubgraphAtomsPackCache>(m,"csubgraphatoms_cache")

  .def_static("subgraphs",[](const Ggraph& G, const Subgraph& S, const int nvecs){
      return ptens_global::c_subgraphatoms_cache(*G.obj,S.obj,nvecs);});


