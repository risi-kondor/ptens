pybind11::class_<BatchedPtensorMap>(m,"batched_ptensor_map")

  .def_static("overlaps_map",[](const BatchedAtomsPack& out, const BatchedAtomsPack& in){
      return BatchedPtensorMap::overlaps_map(out,in);})

  .def_readwrite("atoms",&BatchedPtensorMap::atoms)
  .def_readwrite("out_indices",&BatchedPtensorMap::out_indices)
  .def_readwrite("in_indices",&BatchedPtensorMap::in_indices)

  .def("__str__",[](const BatchedPtensorMap& obj){return obj.str();});

