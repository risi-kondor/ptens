typedef cnine::ATview<float> TVIEW;

pybind11::class_<Ptensors2b<float> >(m,"ptensors2b")

  .def(py::init([](const AtomsPack& atoms, at::Tensor& M){
	return Ptensors2b<float>(atoms,Ltensor<float>(TVIEW(M)));}))
  .def(py::init([](const vector<vector<int> >& atoms, at::Tensor& M){
	return Ptensors2b<float>(AtomsPack(atoms),Ltensor<float>(TVIEW(M)));}))

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors2b<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors2b<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)


  .def("like",[](const Ptensors2b<float>& x, at::Tensor& M){
      return Ptensors2b(x.atoms,ATview<float>(M));})
  .def("copy",[](const Ptensors2b<float>& x){return x.copy();})
  .def("copy",[](const Ptensors2b<float>& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&Ptensors2b<float>::zeros_like)
  .def("randn_like",&Ptensors2b<float>::gaussian_like)


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](Ptensors2b<float>& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
  .def("add_to_grad",[](Ptensors2b<float>& x, const Ptensors2b<float>& y, const float c){x.add_to_grad(y,c);})
  .def("get_grad",[](Ptensors2b<float>& x){return x.get_grad();})

  .def("__getitem__",[](const Ptensors2b<float>& x, const int i){return x(i);})
  .def("torch",[](const Ptensors2b<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors2b<float>::get_dev)
  .def("get_nc",&Ptensors2b<float>::get_nc)
  .def("get_atoms",[](const Ptensors2b<float>& x){return *x.atoms.obj->atoms;})
  .def("dim",&Ptensors2b<float>::dim)

  .def("to_device",&Ptensors2b<float>::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](Ptensors2b<float>& r, const Ptensors2b<float>& x){r.add(x);})
  .def("add_back",[](Ptensors2b<float>& x, const Ptensors2b<float>& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",[](const Ptensors2b<float>& x, const Ptensors2b<float>& y){return cat_channels(x,y);})
  .def("cat_channels_back0",&Ptensors2b<float>::cat_channels_back0)
  .def("cat_channels_back1",&Ptensors2b<float>::cat_channels_back1)

  .def_static("cat",&Ptensors2b<float>::cat)
  .def("add_cat_back",[](Ptensors2b<float>& x, Ptensors2b<float>& r, const int offs){
      x.get_grad()+=r.slices(0,offs,x.dim(0));})

//.def("outer",&Ptensors2b::outer)
//.def("outer_back0",&Ptensors2b::outer_back0)
//.def("outer_back1",&Ptensors2b::outer_back1)

  .def("scale_channels",[](Ptensors2b<float>& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](Ptensors2b<float>& r, const Ptensors2b<float>& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const Ptensors2b<float>& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](Ptensors2b<float>& r, const Ptensors2b<float>& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const Ptensors2b<float>& x, const Ptensors2b<float>& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const Ptensors2b<float>& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](Ptensors2b<float>& r, const Ptensors2b<float>& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const Ptensors2b<float>& x, const Ptensors2b<float>& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const Ptensors2b<float>& x, Ptensors2b<float>& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const Ptensors2b<float>& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",&Ptensors2b<float>::add_ReLU_back)

  .def("inp",[](const Ptensors2b<float>& x, const Ptensors2b<float>& y){return x.inp(y);})
  .def("diff2",[](const Ptensors2b<float>& x, const Ptensors2b<float>& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def("linmaps0",[](const Ptensors2b<float>& x){return linmaps0(x);})
  .def("linmaps1",[](const Ptensors2b<float>& x){return linmaps1(x);})
  .def("linmaps2",[](const Ptensors2b<float>& x){return linmaps2(x);})
  .def("add_linmaps_back",[](Ptensors2b<float>& x, Ptensors0b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors2b<float>& x, Ptensors1b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors2b<float>& x, Ptensors2b<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("gather0",[](const Ptensors2b<float>& x, const AtomsPack& a){return gather0(x,a);})
  .def("gather1",[](const Ptensors2b<float>& x, const AtomsPack& a){return gather1(x,a);})
  .def("gather2",[](const Ptensors2b<float>& x, const AtomsPack& a){return gather2(x,a);})
  .def("add_gather_back",[](Ptensors2b<float>& x, Ptensors0b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](Ptensors2b<float>& x, Ptensors1b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](Ptensors2b<float>& x, Ptensors2b<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors2b<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors2b<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors2b<float>::repr);
