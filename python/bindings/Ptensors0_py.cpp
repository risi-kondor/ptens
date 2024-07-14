pybind11::class_<Ptensors0<float> >(m,"ptensors0")

  .def_static("view",[](const AtomsPack& atoms, at::Tensor& x){
      return Ptensors0<float>(atoms,tensorf::view(x));})


// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](Ptensors0f& obj, const Ptensors0f& x, const TensorLevelMap& tmap){
      return obj.add_gather(x,tmap);}) 
  .def("add_gather",[](Ptensors0f& obj, const Ptensors1f& x, const TensorLevelMap& tmap){
      return obj.add_gather(x,tmap);}) 
  .def("add_gather",[](Ptensors0f& obj, const Ptensors2f& x, const TensorLevelMap& tmap){
      return obj.add_gather(x,tmap);}) 

  .def("add_gather_back",[](Ptensors0f& obj, const Ptensors0f& x, const TensorLevelMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
  .def("add_gather_back",[](Ptensors0f& obj, const Ptensors1f& x, const TensorLevelMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
  .def("add_gather_back",[](Ptensors0f& obj, const Ptensors2f& x, const TensorLevelMap& tmap){
      return obj.add_gather_back(x,tmap);}) 


// ---- Reductions ------------------------------------------------------------------------------------------


  .def("add_reduce0_to",[](Ptensors0f& obj, const Ptensors0f& r, const AindexPack& list){
      obj.add_reduce0_to(r,list);})
  
  .def("broadcast0",[](Ptensors0f& obj, const Ptensors0f& x, const AindexPack& list, const int offs){
      obj.broadcast0(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&Ptensors0<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors0<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0<float>::repr);



