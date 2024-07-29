pybind11::class_<BatchedPtensors1<float> >(m,"batched_ptensors1")

  .def_static("view",[](const BatchedAtomsPackBase& atoms, at::Tensor& x){
      return BatchedPtensors1<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](BPtensors1f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors1f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors1f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](BPtensors1f& obj, const BPtensors0f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](BPtensors1f& obj, const BPtensors1f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](BPtensors1f& obj, const BPtensors2f& x){
      return obj.add_linmaps_back(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](BPtensors1f& obj, const BPtensors0f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](BPtensors1f& obj, const BPtensors1f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](BPtensors1f& obj, const BPtensors2f& x, const BLmap& map){
      return obj.add_gather(x,map);}) 

  .def("add_gather_back",[](BPtensors1f& obj, const BPtensors0f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](BPtensors1f& obj, const BPtensors1f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](BPtensors1f& obj, const BPtensors2f& x, const BLmap& map){
      return obj.add_gather_back(x,map);}) 


// ---- Gather ----------------------------------------------------------------------------------------------


//   .def("add_gather",[](BPtensors1f& obj, const BPtensors0f& x, const BPmap& map){
//       return obj.add_gather(x,map);}) 
//   .def("add_gather",[](BPtensors1f& obj, const BPtensors1f& x, const BPmap& map){
//       return obj.add_gather(x,map);}) 
//   .def("add_gather",[](BPtensors1f& obj, const BPtensors2f& x, const BPmap& map){
//       return obj.add_gather(x,map);}) 

//   .def("add_gather_back",[](BPtensors1f& obj, const BPtensors0f& x, const BPmap& map){
//       return obj.add_gather_back(x,map);}) 
//   .def("add_gather_back",[](BPtensors1f& obj, const BPtensors1f& x, const BPmap& map){
//       return obj.add_gather_back(x,map);}) 
//   .def("add_gather_back",[](BPtensors1f& obj, const BPtensors2f& x, const BPmap& map){
//       return obj.add_gather_back(x,map);}) 


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&BPtensors1f::str,py::arg("indent")="")
  .def("__str__",&BPtensors1f::str,py::arg("indent")="")
  .def("__repr__",&BPtensors1f::repr);



