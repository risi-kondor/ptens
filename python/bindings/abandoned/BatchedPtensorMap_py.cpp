pybind11::class_<BatchedPtensorMap>(m,"batched_ptensor_map")

//.def_static("overlaps",[](const BatchedAtomsPack& out, const BatchedAtomsPack& in){
//    return BatchedPtensorMap::overlaps(out,in);})

  .def_readonly("atoms",&BatchedPtensorMap::_atoms)
  .def_readonly("out_indices",&BatchedPtensorMap::out_indices)
  .def_readonly("in_indices",&BatchedPtensorMap::in_indices)

  .def("__str__",[](const BatchedPtensorMap& obj,const string indent){
      return obj.str();},py::arg("indent")="");
