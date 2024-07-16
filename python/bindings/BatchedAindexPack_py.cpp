pybind11::class_<BatchedAindexPack>(m,"batched_aindexpack")

  .def("__str__",[](const BatchedAindexPack& obj, string indent){return obj.str();},py::arg("indent")=""); 



