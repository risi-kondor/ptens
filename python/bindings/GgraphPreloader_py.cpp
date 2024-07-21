pybind11::class_<ptens::GgraphPreloader>(m,"ggraph_preloader")

  .def(pybind11::init([](const Ggraph& x){
	return GgraphPreloader(x);}))

  .def("__repr__",&GgraphPreloader::repr)
  .def("str",&GgraphPreloader::str,py::arg("indent")="")
  .def("__str__",&GgraphPreloader::str,py::arg("indent")="");


