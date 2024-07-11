pybind11::class_<TensorLevelMap>(m,"tensor_map")

  .def_static("overlaps_map",[](const AtomsPack& out, const AtomsPack& in){
      return TensorLevelMap::overlaps_map(out,in);})

  .def("atoms",&TensorLevelMap::atoms)
  .def("out_indices",&TensorLevelMap::out)
  .def("in_indices",&TensorLevelMap::in)

  .def("__str__",[](const TensorLevelMap& obj){return obj.str();});

