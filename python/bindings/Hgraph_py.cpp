pybind11::class_<Hgraph>(m,"graph")

  .def_static("edge_index",[](const at::Tensor& x){return Hgraph::edge_index(cnine::RtensorA(x));})
  .def_static("matrix",[](const at::Tensor& x){return Hgraph(cnine::RtensorA(x));})
  .def_static("random",static_cast<Hgraph(*)(const int, const float)>(&Hgraph::random))
  .def_static("randomd",static_cast<Hgraph(*)(const int, const float)>(&Hgraph::randomd))

  .def("nhoods",&Hgraph::nhoods)
  .def("set",&Hgraph::set)

  .def("dense",[](const Hgraph& G){return G.dense().torch();})

  .def("subgraphs",[](const Hgraph& G, const Hgraph& H){
      FindPlantedSubgraphs planted(G,H); 
      return AtomsPack(planted.matches);})

  .def("str",&Hgraph::str,py::arg("indent")="")
  .def("__str__",&Hgraph::str,py::arg("indent")="");


