pybind11::class_<Hgraph>(m,"graph")

  .def_static("edge_index",[](const at::Tensor& x, const int n=-1){
      return Hgraph::edge_index(cnine::Ltensor<float>(x),n);})
  .def_static("edge_index",[](const at::Tensor& x, const int n, const int m){
      return Hgraph::edge_index(cnine::Ltensor<float>(x),n,m);})

  .def_static("edge_index",[](const at::Tensor& x, const at::Tensor& l, const int n){
      return Hgraph::edge_index(cnine::Ltensor<float>(x),cnine::Ltensor<float>(l),n);})

  .def_static("matrix",[](const at::Tensor& x){return Hgraph(cnine::Tensor<float>(x));})
  .def_static("matrix",[](const at::Tensor& x, const at::Tensor& l){
      return Hgraph(cnine::Ltensor<float>(x),cnine::Ltensor<float>(l));})

  .def_static("overlaps",[](const AtomsPack& x, const AtomsPack& y){return Hgraph::overlaps(x,y);})
  .def_static("random",static_cast<Hgraph(*)(const int, const float)>(&Hgraph::random))
  .def_static("randomd",static_cast<Hgraph(*)(const int, const float)>(&Hgraph::randomd))

  .def("nhoods",&Hgraph::nhoods)
  .def("edges",&Hgraph::edges)
  .def("set",&Hgraph::set)

  .def("dense",[](const Hgraph& G){return G.dense().torch();})

//.def("subgraphs",[](const Hgraph& G, const Hgraph& H){
//      return AtomsPack(CachedPlantedSubgraphs()(G,H));
//    })

  .def("str",&Hgraph::str,py::arg("indent")="")
  .def("__str__",&Hgraph::str,py::arg("indent")="");


