typedef SubgraphLayer0b<float> SGlayer0b;
typedef SubgraphLayer1b<float> SGlayer1b;
typedef SubgraphLayer2b<float> SGlayer2b;


pybind11::class_<SGlayer0b,Ptensors0b<float> >(m,"subgraphlayer0b")

  .def(pybind11::init<ptens::Ggraph&, const at::Tensor&>())

  .def_static("create",[](const Ggraph& G, const int _nc, const int fcode, const int _dev){
      return SGlayer0b(G,_nc,fcode,_dev);}, 
    py::arg("graph"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("like",[](const SGlayer0b& x, at::Tensor& M){
      return SGlayer0b(x.G,x.S,x.atoms,ATview<float>(M));})
  .def("copy",[](const SGlayer0b& x){return x.copy();})
  .def("copy",[](const SGlayer0b& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&SGlayer0b::zeros_like)
  .def("randn_like",&SGlayer0b::gaussian_like)

  .def("to",[](const SGlayer0b& x, const int dev){return SGlayer0b(x,dev);})
  .def("to_device",[](SGlayer0b& x, const int dev){return SGlayer0b(x,dev);})


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](SGlayer0b& r, const SGlayer0b& x){r.add(x);})

  .def("cat_channels",[](const SGlayer0b& x, const SGlayer0b& y){
      return cat_channels_sg(x,y);})
  .def("cat",&SGlayer0b::cat)
  .def("scale_channels",[](SGlayer0b& x, at::Tensor& y){
      return scale_channels_sg(x,ATview<float>(y));})
  .def("mprod",[](const SGlayer0b& x, at::Tensor& M){
      return mprod_sg(x,ATview<float>(M));})
  .def("linear",[](const SGlayer0b& x, at::Tensor& y, at::Tensor& b){
      return linear_sg(x,ATview<float>(y),ATview<float>(b));})
  .def("ReLU",[](const SGlayer0b& x, const float alpha){
      return ReLU_sg(x,alpha);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const SGlayer0b& x){
      return SGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const SGlayer1b& x){
      return SGlayer1b::linmaps(x);}) 
  .def_static("linmaps",[](const SGlayer2b& x){
      return SGlayer1b::linmaps(x);}) 

  .def(pybind11::init<const SGlayer0b&, const Subgraph&>())
  .def(pybind11::init<const SGlayer1b&, const Subgraph&>())
  .def(pybind11::init<const SGlayer2b&, const Subgraph&>())

  .def(pybind11::init<const Ptensors0b<float>&, const Ggraph&, const Subgraph&>())
  .def(pybind11::init<const Ptensors1b<float>&, const Ggraph&, const Subgraph&>())
  .def(pybind11::init<const Ptensors2b<float>&, const Ggraph&, const Subgraph&>());


// ---- I/O --------------------------------------------------------------------------------------------------

//.def("str",&SGlayer0b::str,py::arg("indent")="")
//.def("__str__",&SGlayer0b::str,py::arg("indent")="")
//.def("__repr__",&SGlayer0b::repr);



//.def("get_dev",&SGlayer0b::get_dev)
//.def("get_nc",&SGlayer0b::get_nc)
//.def("get_atoms",[](const SGlayer0b& x){return x.atoms.as_vecs();})
//.def("linmaps0",[](const SGlayer0b& x){return sglinmaps0(x);})
//.def("linmaps1",[](const SGlayer0b& x){return sglinmaps1(x);})
//.def("linmaps2",[](const SGlayer0b& x){return sglinmaps2(x);})

//.def("gather",[](const SGlayer0b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const SGlayer1b& x, const Subgraph& a){return gather0(x,a);})
//.def("gather",[](const SGlayer2b& x, const Subgraph& a){return gather0(x,a);});
