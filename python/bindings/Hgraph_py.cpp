pybind11::class_<Hgraph>(m,"graph")

  .def_static("matrix",[](const at::Tensor& x){return Hgraph(cnine::RtensorA(x));})
  .def_static("random",static_cast<Hgraph(*)(const int, const float)>(&Hgraph::random))

  .def("nhoods",&Hgraph::nhoods)
  .def("set",&Hgraph::set)

  .def("dense",[](const Hgraph& G){return G.dense().torch();})

  .def("str",&Hgraph::str,py::arg("indent")="")
  .def("__str__",&Hgraph::str,py::arg("indent")="");


