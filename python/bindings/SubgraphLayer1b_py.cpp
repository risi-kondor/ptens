typedef SubgraphLayer0b<float> SGlayer0b;
typedef SubgraphLayer1b<float> SGlayer1b;
typedef SubgraphLayer2b<float> SGlayer2b;


pybind11::class_<SGlayer1b,Ptensors1b<float> >(m,"subgraphlayer1b")

//.def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

  .def_static("create",[](const Ggraph& G, const Subgraph& S, const int _nc, const int fcode, const int _dev){
      return SGlayer1b(G,S,_nc,fcode,_dev);}, 
    py::arg("graph"),py::arg("subgraph"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("like",[](const SGlayer1b& x, at::Tensor& M){
      return SGlayer1b(x.G,x.S,x.atoms,ATview<float>(M));})

  .def("copy",[](const SGlayer1b& x){return x.copy();})
  .def("copy",[](const SGlayer1b& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&SGlayer1b::zeros_like)
  .def("randn_like",&SGlayer1b::gaussian_like)
  .def("to_device",[](SGlayer1b& x, const int dev){return SGlayer1b(x,dev);})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](SGlayer1b& r, const SGlayer1b& x){r.add(x);})

  .def("cat_channels",[](const SGlayer1b& x, const SGlayer1b& y){
      return cat_channels_sg(x,y);})
  .def("cat",&SGlayer1b::cat)
  .def("scale_channels",[](SGlayer1b& x, at::Tensor& y){
      return scale_channels_sg(x,ATview<float>(y));})
  .def("mprod",[](const SGlayer1b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const SGlayer1b& x, at::Tensor& y, at::Tensor& b){
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const SGlayer1b& x, const float alpha){
      return ReLU_sg(x,alpha);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def("linmaps0",[](const SGlayer1b& x){return sglinmaps0(x);})
  .def("linmaps1",[](const SGlayer1b& x){return sglinmaps1(x);})
  .def("linmaps2",[](const SGlayer1b& x){return sglinmaps2(x);})

  .def(pybind11::init<const Ptensors0b<float>&, const Ggraph&, const Subgraph&>())
  .def(pybind11::init<const Ptensors1b<float>&, const Ggraph&, const Subgraph&>())
  .def(pybind11::init<const Ptensors2b<float>&, const Ggraph&, const Subgraph&>())

  .def_static("gather",[](const SGlayer0b& x, const Subgraph& a){return gather1(x,a);})
  .def_static("gather",[](const SGlayer1b& x, const Subgraph& a){return gather1(x,a);})
  .def_static("gather",[](const SGlayer2b& x, const Subgraph& a){return gather1(x,a);});


// ---- I/O --------------------------------------------------------------------------------------------------

//.def("str",&SGlayer0b::str,py::arg("indent")="")
//.def("__str__",&SGlayer0b::str,py::arg("indent")="")
//.def("__repr__",&SGlayer0b::repr);



//.def("get_dev",&SGlayer0b::get_dev)
//.def("get_nc",&SGlayer0b::get_nc)
//.def("get_atoms",[](const SGlayer0b& x){return x.atoms.as_vecs();})
