pybind11::class_<ptens::Ggraph>(m,"ggraph")

  .def(pybind11::init([](const at::Tensor& x){return Ggraph(cnine::Tensor<float>(x));}))

  .def(pybind11::init<const int>())

  .def_static("edge_index",[](const at::Tensor& x, const int n=-1){
      return Ggraph::from_edges(n,cnine::Tensor<float>(x));})

  .def_static("edge_index_cached",[](const int key, const at::Tensor& x, const int n=-1){
      return Ggraph::from_edges(n,cnine::Tensor<float>(x),key);})

  .def_static("random",static_cast<Ggraph(*)(const int, const float)>(&ptens::Ggraph::random))

  .def("dense",[](const Ggraph& G){return G.dense().torch();})

  .def("subgraphs",[](const Ggraph& G, const Subgraph& H){
      return G.subgraphs(H);})

  .def("cache",[](const Ggraph& G, const int key){return G.cache(key);})

  .def("str",&Ggraph::str,py::arg("indent")="")
  .def("__str__",&Ggraph::str,py::arg("indent")="");


