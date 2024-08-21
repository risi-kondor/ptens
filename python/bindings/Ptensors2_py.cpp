pybind11::class_<Ptensors2<float> >(m,"ptensors2")


  .def_static("view",[](const AtomsPack& atoms, at::Tensor& x){
      return Ptensors2<float>(atoms,tensorf::view(x));})


// ---- Linmaps ----------------------------------------------------------------------------------------------


  .def("add_linmaps",[](Ptensors2f& obj, const Ptensors0f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](Ptensors2f& obj, const Ptensors1f& x){
      return obj.add_linmaps(x);}) 
  .def("add_linmaps",[](Ptensors2f& obj, const Ptensors2f& x){
      return obj.add_linmaps(x);}) 
  
  .def("add_linmaps_back",[](Ptensors2f& obj, const Ptensors0f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](Ptensors2f& obj, const Ptensors1f& x){
      return obj.add_linmaps_back(x);}) 
  .def("add_linmaps_back",[](Ptensors2f& obj, const Ptensors2f& x){
      return obj.add_linmaps_back(x);}) 
  

// ---- Gather ----------------------------------------------------------------------------------------------


  .def("add_gather",[](Ptensors2f& obj, const Ptensors0f& x, const LayerMap& tmap){
      return obj.add_gather(x,tmap);}) 
  .def("add_gather",[](Ptensors2f& obj, const Ptensors1f& x, const LayerMap& tmap){
      return obj.add_gather(x,tmap);}) 
  .def("add_gather",[](Ptensors2f& obj, const Ptensors2f& x, const LayerMap& tmap){
      return obj.add_gather(x,tmap);}) 

  .def("add_gather_back",[](Ptensors2f& obj, const Ptensors0f& x, const LayerMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
  .def("add_gather_back",[](Ptensors2f& obj, const Ptensors1f& x, const LayerMap& tmap){
      return obj.add_gather_back(x,tmap);}) 
  .def("add_gather_back",[](Ptensors2f& obj, const Ptensors2f& x, const LayerMap& tmap){
      return obj.add_gather_back(x,tmap);}) 


// ---- Gather ----------------------------------------------------------------------------------------------


//   .def("add_gather",[](Ptensors2f& obj, const Ptensors0f& x, const PtensorMap& tmap){
//       return obj.add_gather(x,tmap);}) 
//   .def("add_gather",[](Ptensors2f& obj, const Ptensors1f& x, const PtensorMap& tmap){
//       return obj.add_gather(x,tmap);}) 
//   .def("add_gather",[](Ptensors2f& obj, const Ptensors2f& x, const PtensorMap& tmap){
//       return obj.add_gather(x,tmap);}) 

//   .def("add_gather_back",[](Ptensors2f& obj, const Ptensors0f& x, const PtensorMap& tmap){
//       return obj.add_gather_back(x,tmap);}) 
//   .def("add_gather_back",[](Ptensors2f& obj, const Ptensors1f& x, const PtensorMap& tmap){
//       return obj.add_gather_back(x,tmap);}) 
//   .def("add_gather_back",[](Ptensors2f& obj, const Ptensors2f& x, const PtensorMap& tmap){
//       return obj.add_gather_back(x,tmap);}) 


// ---- Reductions ------------------------------------------------------------------------------------------

/*
  .def("add_reduce0_to",[](const Ptensors2f& obj, at::Tensor& r){
      obj.add_reduce0_to(tensorf::view(r));},py::arg("r"))
  .def("add_reduce1_to",[](const Ptensors2f& obj, at::Tensor& r){
      obj.add_reduce1_to(tensorf::view(r));},py::arg("r"))

  .def("add_reduce0_shrink_to",[](const Ptensors2f& obj, at::Tensor& r, const int offs){
      obj.add_reduce0_shrink_to(tensorf::view(r),offs);},py::arg("r"),py::arg("offs")=0)
  .def("add_reduce1_shrink_to",[](const Ptensors2f& obj, at::Tensor& r, const int offs){
      obj.add_reduce1_shrink_to(tensorf::view(r),offs);},py::arg("r"),py::arg("offs")=0)
  .def("add_reduce2_shrink_to",[](const Ptensors2f& obj, at::Tensor& r, const int offs){
      obj.add_reduce2_shrink_to(tensorf::view(r),offs);},py::arg("r"),py::arg("offs")=0)
*/

// ---- Broadcasting ----------------------------------------------------------------------------------------

/*
  .def("broadcast0",[](Ptensors2f& obj, const at::Tensor& x, const int offs){
      obj.broadcast0(tensorf::view(x),offs);},py::arg("x"),py::arg("offs")=0)
  .def("broadcast1",[](Ptensors2f& obj, const at::Tensor& x, const int offs){
      obj.broadcast1(tensorf::view(x),offs);},py::arg("x"),py::arg("offs")=0)
  .def("broadcast2",[](Ptensors2f& obj, const at::Tensor& x, const int offs){
      obj.broadcast2(tensorf::view(x),offs);},py::arg("x"),py::arg("offs")=0)

  .def("broadcast0_shrink",[](Ptensors2f& obj, const at::Tensor& x){
      obj.broadcast0_shrink(tensorf::view(x));},py::arg("x"))
  .def("broadcast1_shrink",[](Ptensors2f& obj, const at::Tensor& x){
      obj.broadcast1_shrink(tensorf::view(x));},py::arg("x"))
*/

// ---- Indexed Reductions -----------------------------------------------------------------------------------


/*
  .def("add_reduce0_to",[](const Ptensors2f& obj, const Ptensors0f& r, const AindexPack& list){
      obj.add_reduce0_to(r,list);})
  .def("add_reduce1_to",[](const Ptensors2f& obj, const Ptensors1f& r, const AindexPack& list){
      obj.add_reduce1_to(r,list);})
  .def("add_reduce2_to",[](const Ptensors2f& obj, const Ptensors2f& r, const AindexPack& list){
      obj.add_reduce2_to(r,list);})
  
  .def("add_reduce0_shrink_to",[](const Ptensors2f& obj, const Ptensors0f& r, const AindexPack& list){
      obj.add_reduce0_to(r,list);})
  .def("add_reduce1_shrink_to",[](const Ptensors2f& obj, const Ptensors1f& r, const AindexPack& list){
      obj.add_reduce1_to(r,list);})
  .def("add_reduce2_shrink_to",[](const Ptensors2f& obj, const Ptensors2f& r, const AindexPack& list){
      obj.add_reduce2_to(r,list);})
*/

  
// ---- Indexed Broadcasting ---------------------------------------------------------------------------------


/*
  .def("broadcast0",[](Ptensors2f& obj, const Ptensors0f& x, const AindexPack& list, const int offs){
      obj.broadcast0(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)
  .def("broadcast1",[](Ptensors2f& obj, const Ptensors1f& x, const AindexPack& list, const int offs){
      obj.broadcast1(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)
  .def("broadcast2",[](Ptensors2f& obj, const Ptensors2f& x, const AindexPack& list, const int offs){
      obj.broadcast2(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)

  .def("broadcast0_shrink",[](Ptensors2f& obj, const Ptensors0f& x, const AindexPack& list, const int offs){
      obj.broadcast0(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)
  .def("broadcast1_shrink",[](Ptensors2f& obj, const Ptensors1f& x, const AindexPack& list, const int offs){
      obj.broadcast1(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)
*/


// ---- I/O ------------------------------------------------------------------------------------------


  .def("str",&Ptensors2<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors2<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors2<float>::repr);



