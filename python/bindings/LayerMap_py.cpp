pybind11::class_<LayerMap>(m,"layer_map")

  .def_static("overlaps_map",[](const AtomsPack& out, const AtomsPack& in){
      return LayerMap::overlaps_map(out,in);})

  .def_static("overlaps_map",[](const CompressedAtomsPack& out, const CompressedAtomsPack& in){
      return LayerMap::overlaps_map(out.atoms(),in.atoms());})


  .def("__repr__",[](const LayerMap& obj){return obj.repr();})
  .def("__str__",[](const LayerMap& obj, const string indent){return obj.str();},py::arg("indent")="");

