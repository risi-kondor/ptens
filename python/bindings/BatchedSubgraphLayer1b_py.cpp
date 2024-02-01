typedef BatchedSubgraphLayer0b<float> BSGlayer0b;
typedef BatchedSubgraphLayer1b<float> BSGlayer1b;
typedef BatchedSubgraphLayer2b<float> BSGlayer2b;


pybind11::class_<BSGlayer1b,BatchedPtensors1b<float> >(m,"batched_subgraphlayer1b")

//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

//.def_static("create",[](const BatchedGgraph& G, const int _nc, const int fcode, const int _dev){
//    return BSGlayer1b(G,_nc,fcode,_dev);}, 
//  py::arg("graph"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("from_edge_features",[](const vector<int>& keys, at::Tensor&M){
      return BSGlayer1b::from_edge_features(keys,ATview<float>(M));})

  .def_static("like",[](const BSGlayer1b& x, at::Tensor& M){
      return BSGlayer1b(x.G,x.S,x.atoms,ATview<float>(M));})

  .def("copy",[](const BSGlayer1b& x){return x.copy();})
  .def("copy",[](const BSGlayer1b& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const BSGlayer1b& x){return x.zeros_like();})
  .def("randn_like",&BSGlayer1b::gaussian_like)

//.def("to",[](const BSGlayer1b& x, const int dev){return BSGlayer1b(x,dev);})
//.def("to_device",[](BSGlayer1b& x, const int dev){return BSGlayer1b(x,dev);})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](BSGlayer1b& r, const BSGlayer1b& x){r.add(x);})

  .def("cat_channels",[](const BSGlayer1b& x, const BSGlayer1b& y){
      return cat_channels_sg(x,y);})
//.def("cat",&BSGlayer1b::cat)
//.def("scale_channels",[](BSGlayer1b& x, at::Tensor& y){
//      return scale_channels_sg(x,ATview<float>(y));})
  .def("mprod",[](const BSGlayer1b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const BSGlayer1b& x, at::Tensor& y, at::Tensor& b){
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const BSGlayer1b& x, const float alpha){
      return ReLU_sg(x,alpha);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const BSGlayer1b& x){return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer1b& x){return BSGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const BSGlayer2b& x){return BSGlayer1b::linmaps(x);}) 

  .def(pybind11::init<const BSGlayer0b&, const Subgraph&>())
  .def(pybind11::init<const BSGlayer1b&, const Subgraph&>())
  .def(pybind11::init<const BSGlayer2b&, const Subgraph&>())

  .def(pybind11::init<const BatchedPtensors0b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors1b<float>&, const BatchedGgraph&, const Subgraph&>())
  .def(pybind11::init<const BatchedPtensors2b<float>&, const BatchedGgraph&, const Subgraph&>())

  .def("autobahn",[](const BSGlayer1b& x, at::Tensor& W, at::Tensor& B){
      return x.autobahn(ATview<float>(W),ATview<float>(B));})
  .def("add_autobahn_back0",[](BSGlayer1b& x, BSGlayer1b& r, at::Tensor& W){
      x.add_autobahn_back0(r.get_grad(),ATview<float>(W));})
  .def("autobahn_back1",[](BSGlayer1b& x, at::Tensor& W, at::Tensor& B, BSGlayer1b& r){
      x.add_autobahn_back1_to(ATview<float>(W), ATview<float>(B),r.get_grad());});


// ---- I/O --------------------------------------------------------------------------------------------------

//.def("str",&BSGlayer1b::str,py::arg("indent")="")
//.def("__str__",&BSGlayer1b::str,py::arg("indent")="")
//.def("__repr__",&BSGlayer1b::repr);



//.def("get_dev",&BSGlayer1b::get_dev)
//.def("get_nc",&BSGlayer1b::get_nc)
//.def("get_atoms",[](const BSGlayer1b& x){return x.atoms.as_vecs();})
//.def("linmaps0",[](const BSGlayer1b& x){return sglinmaps0(x);})
//.def("linmaps1",[](const BSGlayer1b& x){return sglinmaps1(x);})
//.def("linmaps2",[](const BSGlayer1b& x){return sglinmaps2(x);})

//.def("gather",[](const BSGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const BSGlayer2b& x, const Subgraph& a){return gather0(x,a);});
