pybind11::class_<ptens::Ggraph>(m,"ggraph")


//.def(pybind11::init([](const at::Tensor& x){
//return Ggraph(cnine::Ltensor<float>(x));}))


  .def_static("from_matrix",[](const at::Tensor& x){
	return Ggraph(cnine::Ltensor<float>(x));})

  .def_static("from_matrix",[](const at::Tensor& x, const at::Tensor& labels){
	return Ggraph(cnine::Ltensor<float>(x),cnine::Ltensor<float>(labels));})

  .def_static("from_edge_index",[](const at::Tensor& x, const int n=-1){
      return Ggraph::from_edges(n,cnine::Tensor<float>(x));})

  .def_static("from_edge_index",[](const at::Tensor& x, const at::Tensor& labels, const int n=-1){
      auto G=Ggraph::from_edges(n,cnine::Tensor<float>(x));
      G.set_labels(labels);
    })

  .def_static("random",static_cast<Ggraph(*)(const int, const float)>(&ptens::Ggraph::random))


// ---- Caching ----------------------------------------------------------------------------------------------


  .def("cache",[](const Ggraph& G, const int key){return G.cache(key);})

  .def_static("from_cache",[](const int i){
      return Ggraph(i);})


// ---- Access -----------------------------------------------------------------------------------------------


  .def("dense",[](const Ggraph& G){return G.dense().torch();})


  .def("is_labeled",[](Ggraph& G){
      return G.is_labeled();})
  .def("set_labels",[](Ggraph& G, const at::Tensor& x){
      G.set_labels(cnine::Ltensor<float>(x));})
  .def("get_labels",[](Ggraph& G){
      return G.get_labels().torch();})

  .def("subgraphs",[](const Ggraph& G, const Subgraph& H){
      return G.subgraphs(H);})
  .def("cached_subgraph_lists_as_map",[](const Ggraph& G){
      return G.cached_subgraph_lists_as_map();})

//.def("csubgraphs",[](const Ggraph& G, const Subgraph& H, const int nvecs){
//    return G.compressed_subgraphs(H,nvecs);})

  .def("str",&Ggraph::str,py::arg("indent")="")
  .def("__str__",&Ggraph::str,py::arg("indent")="");


