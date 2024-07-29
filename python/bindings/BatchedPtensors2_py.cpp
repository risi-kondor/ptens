pybind11::class_<BatchedPtensors2<float> >(m,"batched_ptensors2")

  .def_static("view",[](const BatchedAtomsPackBase& atoms, at::Tensor& x){
      return BatchedPtensors2<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](BPtensors2f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors2f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors2f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](BPtensors2f& obj, const BPtensors0f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](BPtensors2f& obj, const BPtensors1f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](BPtensors2f& obj, const BPtensors2f& x){
      return obj.add_linmaps_back(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](BPtensors2f& obj, const BPtensors0f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](BPtensors2f& obj, const BPtensors1f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](BPtensors2f& obj, const BPtensors2f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 

  .def("add_gather_back",[](BPtensors2f& obj, const BPtensors0f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](BPtensors2f& obj, const BPtensors1f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](BPtensors2f& obj, const BPtensors2f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&BPtensors2f::str,py::arg("indent")="")
  .def("__str__",&BPtensors2f::str,py::arg("indent")="")
  .def("__repr__",&BPtensors2f::repr);



