typedef cnine::ATview<float> TVIEW;

pybind11::class_<Ptensors0b<float> >(m,"ptensors0b")

  .def(py::init([](at::Tensor& M){
	return Ptensors0b(Ltensor<float>(TVIEW(M)));}))
  .def(py::init([](const AtomsPack& atoms, at::Tensor& M){
	return Ptensors0b(atoms,Ltensor<float>(TVIEW(M)));}))
  .def(py::init([](const vector<vector<int> >& atoms, at::Tensor& M){
	return Ptensors0b(AtomsPack(atoms),Ltensor<float>(TVIEW(M)));}))

  .def_static("create",[](const int n, const int _nc, const int fcode, const int _dev){
      return Ptensors0b<float>(n,_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors0b<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors0b<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def("copy",[](const Ptensors0b<float>& x){return x.copy();})
  .def("copy",[](const Ptensors0b<float>& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&Ptensors0b<float>::zeros_like)
  .def("randn_like",&Ptensors0b<float>::gaussian_like)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors0b<float>& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
  .def("add_to_grad",[](Ptensors0b<float>& x, const Ptensors0b<float>& y, const float c){x.add_to_grad(y,c);})
  .def("get_grad",[](Ptensors0b<float>& x){return x.get_grad();})

  .def("__getitem__",[](const Ptensors0b<float>& x, const int i){return x(i);})
  .def("torch",[](const Ptensors0b<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors0b<float>::get_dev)
  .def("get_nc",&Ptensors0b<float>::get_nc)
  .def("get_atoms",[](const Ptensors0b<float>& x){return *x.atoms.obj->atoms;})
  .def("dim",&Ptensors0b<float>::dim)

  .def("to_device",&Ptensors0b<float>::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](PtensorLayer<float>& r, const PtensorLayer<float>& x){r.add(x);})
  .def("add_back",[](PtensorLayer<float>& x, const PtensorLayer<float>& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",[](const Ptensors0b<float>& x, const Ptensors0b<float>& y){return cat_channels(x,y);})
  .def("cat_channels_back0",&Ptensors0b<float>::cat_channels_back0)
  .def("cat_channels_back1",&Ptensors0b<float>::cat_channels_back1)

//.def_static("cat",&Ptensors0b::cat)
//.def("add_cat_back",[](Ptensors0b& x, Ptensors0b& r, const int offs){
//    x.get_grad()+=r.slices(0,offs,x.dim(0));})

//.def("outer",&Ptensors0b::outer)
//.def("outer_back0",&Ptensors0b::outer_back0)
//.def("outer_back1",&Ptensors0b::outer_back1)

  .def("scale_channels",[](Ptensors0b<float>& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](Ptensors0b<float>& r, const Ptensors0b<float>& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const Ptensors0b<float>& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](Ptensors0b<float>& r, const Ptensors0b<float>& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const Ptensors0b<float>& x, const Ptensors0b<float>& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const Ptensors0b<float>& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](Ptensors0b<float>& r, const Ptensors0b<float>& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const Ptensors0b<float>& x, const Ptensors0b<float>& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const Ptensors0b<float>& x, Ptensors0b<float>& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const Ptensors0b<float>& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",&Ptensors0b<float>::add_ReLU_back)

  .def("inp",[](const Ptensors0b<float>& x, const Ptensors0b<float>& y){return x.inp(y);})
  .def("diff2",[](const Ptensors0b<float>& x, const Ptensors0b<float>& y){return x.diff2(y);})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors0b<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors0b<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0b<float>::repr);
