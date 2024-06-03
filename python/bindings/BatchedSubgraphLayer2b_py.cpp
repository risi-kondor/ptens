typedef BatchedSubgraphLayer0b<float> BSGlayer0b;
typedef BatchedSubgraphLayer1b<float> BSGlayer1b;
typedef BatchedSubgraphLayer2b<float> BSGlayer2b;


pybind11::class_<BSGlayer2b,BatchedPtensors2b<float> >(m,"batched_subgraphlayer2b")

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


  .def_static("linmaps",[](const BSGlayer0b& x){
      return BSGlayer2b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer1b& x){
      return BSGlayer2b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer2b& x){
      return BSGlayer2b::linmaps(x);}) 

  .def(pybind11::init<const BSGlayer0b&, const Subgraph&>())
  .def(pybind11::init<const BSGlayer1b&, const Subgraph&>())
  .def(pybind11::init<const BSGlayer2b&, const Subgraph&>())

  .def(pybind11::init<const BatchedPtensors0b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors1b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors2b<float>&, const BatchedGgraph&, const Subgraph&>())


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&BSGlayer2b::str,py::arg("indent")="")
  .def("__str__",&BSGlayer2b::str,py::arg("indent")="")
  .def("__repr__",&BSGlayer2b::repr);
