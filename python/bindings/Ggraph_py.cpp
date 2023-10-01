pybind11::class_<ptens::Ggraph>(m,"ggraph")

  .def(pybind11::init<const at::Tensor&>())

  .def_static("edge_index",[](const at::Tensor& x, const int n=-1){
      return Ggraph::edges(n,cnine::RtensorA(x));})

  .def_static("random",static_cast<Ggraph(*)(const int, const float)>(&ptens::Ggraph::random))

  .def("dense",[](const Ggraph& G){return G.dense().torch();})

  .def("str",&Ggraph::str,py::arg("indent")="")
  .def("__str__",&Ggraph::str,py::arg("indent")="");


