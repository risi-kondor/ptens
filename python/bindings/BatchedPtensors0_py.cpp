pybind11::class_<BatchedPtensors0<float> >(m,"batched_ptensors0")

  .def_static("view",[](const BatchedAtomsPackBase& atoms, at::Tensor& x){
      return BatchedPtensors0<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](BPtensors0f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors0f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors0f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](BPtensors0f& obj, const BPtensors0f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](BPtensors0f& obj, const BPtensors1f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](BPtensors0f& obj, const BPtensors2f& x){
      return obj.add_linmaps_back(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](BPtensors0f& obj, const BPtensors0f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](BPtensors0f& obj, const BPtensors1f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](BPtensors0f& obj, const BPtensors2f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 

  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors0f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors1f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors2f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 


// ---- Gather ----------------------------------------------------------------------------------------------


//   .def("add_gather",[](BPtensors0f& obj, const BPtensors0f& x, const BPmap& map){
//       return obj.add_gather(x,map);}) 
//   .def("add_gather",[](BPtensors0f& obj, const BPtensors1f& x, const BPmap& map){
//       return obj.add_gather(x,map);}) 
//   .def("add_gather",[](BPtensors0f& obj, const BPtensors2f& x, const BPmap& map){
//       return obj.add_gather(x,map);}) 

//   .def("add_gather_back",[](BPtensors0f& obj, const BPtensors0f& x, const BPmap& map){
//       return obj.add_gather_back(x,map);}) 
//   .def("add_gather_back",[](BPtensors0f& obj, const BPtensors1f& x, const BPmap& map){
//       return obj.add_gather_back(x,map);}) 
//   .def("add_gather_back",[](BPtensors0f& obj, const BPtensors2f& x, const BPmap& map){
//       return obj.add_gather_back(x,map);}) 


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&BPtensors0f::str,py::arg("indent")="")
  .def("__str__",&BPtensors0f::str,py::arg("indent")="")
  .def("__repr__",&BPtensors0f::repr);

