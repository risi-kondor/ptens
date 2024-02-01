typedef BatchedSubgraphLayer0b<float> BSGlayer0b;
typedef BatchedSubgraphLayer1b<float> BSGlayer1b;
typedef BatchedSubgraphLayer2b<float> BSGlayer2b;


pybind11::class_<BSGlayer2b,BatchedPtensors2b<float> >(m,"batched_subgraphlayer2b")

//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

//.def_static("create",[](const BatchedGgraph& G, const int _nc, const int fcode, const int _dev){
//    return BSGlayer2b(G,_nc,fcode,_dev);}, 
//  py::arg("graph"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("like",[](const BSGlayer2b& x, at::Tensor& M){
      return BSGlayer2b(x.G,x.S,x.atoms,ATview<float>(M));})
  .def("copy",[](const BSGlayer2b& x){return x.copy();})
  .def("copy",[](const BSGlayer2b& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&BSGlayer2b::zeros_like)
  .def("randn_like",&BSGlayer2b::gaussian_like)

//.def("to",[](const BSGlayer2b& x, const int dev){return BSGlayer2b(x,dev);})
//.def("to_device",[](BSGlayer2b& x, const int dev){return BSGlayer2b(x,dev);})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BSGlayer2b& r, const BSGlayer2b& x){r.add(x);})

  .def("cat_channels",[](const BSGlayer2b& x, const BSGlayer2b& y){return cat_channels_sg(x,y);})
//.def("cat",&BSGlayer2b::cat)
//.def("scale_channels",[](BSGlayer2b& x, at::Tensor& y){
//      return scale_channels_sg(x,ATview<float>(y));})
  .def("mprod",[](const BSGlayer2b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const BSGlayer2b& x, at::Tensor& y, at::Tensor& b){
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const BSGlayer2b& x, const float alpha){
      return ReLU_sg(x,alpha);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BSGlayer2b& x){return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer1b& x){return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer2b& x){return BSGlayer1b::linmaps(x);}) 

  .def(pybind11::init<const BSGlayer0b&, const Subgraph&>())
  .def(pybind11::init<const BSGlayer1b&, const Subgraph&>())
  .def(pybind11::init<const BSGlayer2b&, const Subgraph&>())

  .def(pybind11::init<const BatchedPtensors0b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors1b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors2b<float>&, const BatchedGgraph&, const Subgraph&>());


// ---- I/O --------------------------------------------------------------------------------------------------

//.def("str",&BSGlayer2b::str,py::arg("indent")="")
//.def("__str__",&BSGlayer2b::str,py::arg("indent")="")
//.def("__repr__",&BSGlayer2b::repr);



//.def("get_dev",&BSGlayer2b::get_dev)
//.def("get_nc",&BSGlayer2b::get_nc)
//.def("get_atoms",[](const BSGlayer2b& x){return x.atoms.as_vecs();})
//.def("linmaps0",[](const BSGlayer2b& x){return sglinmaps0(x);})
//.def("linmaps1",[](const BSGlayer2b& x){return sglinmaps1(x);})
//.def("linmaps2",[](const BSGlayer2b& x){return sglinmaps2(x);})

//.def("gather",[](const BSGlayer2b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer2b& x, const Subgraph& a){return gather0(x,a);});
