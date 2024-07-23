pybind11::class_<BatchedLayerMap>(m,"batched_layer_map")

  .def_static("overlaps_map",[](const BatchedAtomsPackBase& out, const BatchedAtomsPackBase& in){
      return BatchedLayerMap::overlaps_map(out,in);})

//   .def_static("overlaps_map",[](const BatchedAtomsPack<0>& out, const BatchedAtomsPack<0>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<0>& out, const BatchedAtomsPack<1>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<0>& out, const BatchedAtomsPack<2>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<1>& out, const BatchedAtomsPack<0>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<1>& out, const BatchedAtomsPack<1>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<1>& out, const BatchedAtomsPack<2>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<2>& out, const BatchedAtomsPack<0>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<2>& out, const BatchedAtomsPack<1>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})
//   .def_static("overlaps_map",[](const BatchedAtomsPack<2>& out, const BatchedAtomsPack<2>& in){
//       return BatchedLayerMap::overlaps_map(out,in);})

  .def("__getitem__",&BatchedLayerMap::operator[])

  .def("__repr__",[](const BatchedLayerMap& obj){return obj.repr();})
  .def("__str__",[](const BatchedLayerMap& obj, const string indent){return obj.str();},py::arg("indent")="");

