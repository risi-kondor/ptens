pybind11::class_<BatchedPtensors0<float> >(m,"batched_ptensors0")

  .def_static("view",[](const BatchedAtomsPack& atoms, at::Tensor& x){
      return BatchedPtensors0<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](BPtensors0f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors0f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](BPtensors0f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](BPtensors0f& obj, const BPtensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps_back",[](BPtensors0f& obj, const BPtensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps_back",[](BPtensors0f& obj, const BPtensors2f& x){
      return obj.add_linmaps(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](BPtensors0f& obj, const BPtensors0f& x){
      return obj.add_gather(x);}) 
  .def("add_gather",[](BPtensors0f& obj, const BPtensors1f& x){
      return obj.add_gather(x);}) 
  .def("add_gather",[](BPtensors0f& obj, const BPtensors2f& x){
      return obj.add_gather(x);}) 

  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors0f& x){
      return obj.add_gather_back(x);}) 
  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors1f& x){
      return obj.add_gather_back(x);}) 
  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors2f& x){
      return obj.add_gather_back(x);}) 


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&BPtensors0f::str,py::arg("indent")="")
  .def("__str__",&BPtensors0f::str,py::arg("indent")="")
  .def("__repr__",&BPtensors0f::repr);



/*
  .def("add_gather",[](BPtensors0f& obj, const BPtensors0f& x, const TensorLevelMap& tmap){
      return obj.add_gather(x,tmap);}) 
  .def("add_gather",[](BPtensors0f& obj, const BPtensors1f& x, const TensorLevelMap& tmap){
      return obj.add_gather(x,tmap);}) 
  .def("add_gather",[](BPtensors0f& obj, const BPtensors2f& x, const TensorLevelMap& tmap){
      return obj.add_gather(x,tmap);}) 

  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors0f& x, const TensorLevelMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors1f& x, const TensorLevelMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
  .def("add_gather_back",[](BPtensors0f& obj, const BPtensors2f& x, const TensorLevelMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
*/
