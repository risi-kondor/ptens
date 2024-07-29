pybind11::class_<Ptensors1<float> >(m,"ptensors1")

  .def_static("view",[](const AtomsPack& atoms, at::Tensor& x){
      return Ptensors1<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](Ptensors1f& obj, const Ptensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](Ptensors1f& obj, const Ptensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](Ptensors1f& obj, const Ptensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](Ptensors1f& obj, const Ptensors0f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](Ptensors1f& obj, const Ptensors1f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](Ptensors1f& obj, const Ptensors2f& x){
      return obj.add_linmaps_back(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](Ptensors1f& obj, const Ptensors0f& x, const LayerMap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](Ptensors1f& obj, const Ptensors1f& x, const LayerMap& map){
      return obj.add_gather(x,map);}) 
  .def("add_gather",[](Ptensors1f& obj, const Ptensors2f& x, const LayerMap& map){
      return obj.add_gather(x,map);}) 

  .def("add_gather_back",[](Ptensors1f& obj, const Ptensors0f& x, const LayerMap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](Ptensors1f& obj, const Ptensors1f& x, const LayerMap& map){
      return obj.add_gather_back(x,map);}) 
  .def("add_gather_back",[](Ptensors1f& obj, const Ptensors2f& x, const LayerMap& map){
      return obj.add_gather_back(x,map);}) 


// ---- Gather ----------------------------------------------------------------------------------------------


//   .def("add_gather",[](Ptensors1f& obj, const Ptensors0f& x, const PtensorMap& tmap){
//       return obj.add_gather(x,tmap);}) 
//   .def("add_gather",[](Ptensors1f& obj, const Ptensors1f& x, const PtensorMap& tmap){
//       return obj.add_gather(x,tmap);}) 
//   .def("add_gather",[](Ptensors1f& obj, const Ptensors2f& x, const PtensorMap& tmap){
//       return obj.add_gather(x,tmap);}) 

//   .def("add_gather_back",[](Ptensors1f& obj, const Ptensors0f& x, const PtensorMap& tmap){
//       return obj.add_gather_back(x,tmap);}) 
//   .def("add_gather_back",[](Ptensors1f& obj, const Ptensors1f& x, const PtensorMap& tmap){
//       return obj.add_gather_back(x,tmap);}) 
//   .def("add_gather_back",[](Ptensors1f& obj, const Ptensors2f& x, const PtensorMap& tmap){
//       return obj.add_gather_back(x,tmap);}) 


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&Ptensors1<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors1<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors1<float>::repr);




// ---- Reductions ------------------------------------------------------------------------------------------


//  .def("add_reduce0_to",[](const Ptensors1f& obj, at::Tensor& r, const int offs){
//      obj.add_reduce0_to(tensorf::view(r),offs);},py::arg("r"),py::arg("offs")=0)

//  .def("add_reduce0_to",[](const Ptensors1f& obj, const Ptensors0f& r, const AindexPack& list){
//      obj.add_reduce0_to(r,list);})
//  .def("add_reduce1_to",[](const Ptensors1f& obj, const Ptensors1f& r, const AindexPack& list){
//      obj.add_reduce1_to(r,list);})

  
// ---- Broadcasting ----------------------------------------------------------------------------------------


//   .def("broadcast0",[](Ptensors1f& obj, const at::Tensor& x, const int offs){
//       obj.broadcast0(tensorf::view(x),offs);},py::arg("x"),py::arg("offs")=0)

//   .def("broadcast0",[](Ptensors1f& obj, const Ptensors0f& x, const AindexPack& list, const int offs){
//       obj.broadcast0(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)
//   .def("broadcast1",[](Ptensors1f& obj, const Ptensors1f& x, const AindexPack& list, const int offs){
//       obj.broadcast1(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)


