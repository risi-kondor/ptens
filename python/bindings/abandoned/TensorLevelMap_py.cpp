pybind11::class_<PtensorMap>(m,"tensor_map")

//.def_static("overlaps_map",[](const AtomsPack& out, const AtomsPack& in){
//    return PtensorMap::overlaps_map(out,in);})

  .def("atoms",&PtensorMap::atoms)
  .def("out_indices",&PtensorMap::out)
  .def("in_indices",&PtensorMap::in)

  .def("__str__",[](const PtensorMap& obj){return obj.str();});

