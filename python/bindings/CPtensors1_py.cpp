pybind11::class_<CompressedPtensors1<float> >(m,"cptensors1")

  .def_static("view",[](const CompressedAtomsPack& atoms, at::Tensor& x){
      return CPtensors1f(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](CPtensors1f& obj, const Ptensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](CPtensors1f& obj, const CPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](CPtensors1f& obj, const CPtensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](CPtensors1f& obj, const Ptensors0f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](CPtensors1f& obj, const CPtensors1f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](CPtensors1f& obj, const CPtensors2f& x){
      return obj.add_linmaps_back(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](CPtensors1f& obj, const Ptensors0f& x, const LayerMap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](CPtensors1f& obj, const CPtensors1f& x, const LayerMap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](CPtensors1f& obj, const CPtensors2f& x, const LayerMap& map){
      return obj.add_gather(x,map);}) 

  .def("add_gather_back",[](CPtensors1f& obj, const Ptensors0f& x, const LayerMap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](CPtensors1f& obj, const CPtensors1f& x, const LayerMap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](CPtensors1f& obj, const CPtensors2f& x, const LayerMap& map){
      return obj.add_gather_back(x,map);})


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&CompressedPtensors1<float>::str,py::arg("indent")="")
  .def("__str__",&CompressedPtensors1<float>::str,py::arg("indent")="")
  .def("__repr__",&CompressedPtensors1<float>::repr);

