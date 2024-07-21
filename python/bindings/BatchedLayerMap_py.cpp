pybind11::class_<BatchedLayerMap>(m,"batched_layer_map")

  .def_static("overlaps_map",[](const BatchedAtomsPack& out, const BatchedAtomsPack& in){
      return BatchedLayerMap::overlaps_map(out,in);})

  .def("__getitem__",&BatchedLayerMap::operator[])

  .def("__repr__",[](const BatchedLayerMap& obj){return obj.repr();})
  .def("__str__",[](const BatchedLayerMap& obj, const string indent){return obj.str();},py::arg("indent")="");

