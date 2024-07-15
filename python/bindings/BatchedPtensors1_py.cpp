pybind11::class_<BatchedPtensors1<float> >(m,"batched_ptensors1")

  .def_static("view",[](const BatchedAtomsPack& atoms, at::Tensor& x){
      return BatchedPtensors1<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](BPtensors1f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors1f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors1f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](BPtensors1f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps_back",[](BPtensors1f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps_back",[](BPtensors1f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](BPtensors1f& obj, const BPtensors0f& x){
      return obj.add_gather(x);}) 
  .def("add_gather",[](BPtensors1f& obj, const BPtensors1f& x){
      return obj.add_gather(x);}) 
  .def("add_gather",[](BPtensors1f& obj, const BPtensors2f& x){
      return obj.add_gather(x);}) 

  .def("add_gather_back",[](BPtensors1f& obj, const BPtensors0f& x){
      return obj.add_gather_back(x);}) 
  .def("add_gather_back",[](BPtensors1f& obj, const BPtensors1f& x){
      return obj.add_gather_back(x);}) 
  .def("add_gather_back",[](BPtensors1f& obj, const BPtensors2f& x){
      return obj.add_gather_back(x);}) 


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&BPtensors1f::str,py::arg("indent")="")
  .def("__str__",&BPtensors1f::str,py::arg("indent")="")
  .def("__repr__",&BPtensors1f::repr);



