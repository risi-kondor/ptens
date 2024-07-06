typedef cnine::ATview<float> TVIEW;

pybind11::class_<Ptensors2<float> >(m,"ptensors2")

  .def(py::init([](const AtomsPack& atoms, at::Tensor& M){
	return Ptensors2<float>(atoms,Ltensor<float>(TVIEW(M)));}))
  .def(py::init([](const vector<vector<int> >& atoms, at::Tensor& M){
	return Ptensors2<float>(AtomsPack(atoms),Ltensor<float>(TVIEW(M)));}))

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors2<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors2<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)


//  .def("like",[](const Ptensors2<float>& x, at::Tensor& M){
//      return Ptensors2<float>(x.atoms,ATview<float>(M));})
  .def("copy",[](const Ptensors2<float>& x){return x.copy();})
  .def("copy",[](const Ptensors2<float>& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const Ptensors2<float>& x){return Ptensors2<float>::zeros_like(x);})
  .def("randn_like",[](const Ptensors2<float>& x){return Ptensors2<float>::gaussian_like(x);})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("__len__",&Ptensors2<float>::size)
//.def("add_to_grad",[](Ptensors2<float>& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
//.def("add_to_grad",[](Ptensors2<float>& x, const Ptensors2<float>& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](Ptensors2<float>& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
  .def("add_to_grad",[](Ptensors2<float>& x, const Ptensors2<float>& y, const float c){x.get_grad().add(y,c);})
  .def("get_grad",[](Ptensors2<float>& x){return x.get_grad();})

  .def("__getitem__",[](const Ptensors2<float>& x, const int i){return x(i);})
  .def("torch",[](const Ptensors2<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&Ptensors2<float>::get_dev)
  .def("get_nc",&Ptensors2<float>::get_nc)
  .def("get_atoms",[](const Ptensors2<float>& x){return x.get_atoms();})
  .def("dim",&Ptensors2<float>::dim)

  .def("to",[](const Ptensors2<float>& x, const int dev){return Ptensors2<float>(x,dev);})
  .def("to_device",&Ptensors2<float>::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


  .def("add",[](Ptensors2<float>& r, const Ptensors2<float>& x){r.add(x);})
  .def("add_back",[](Ptensors2<float>& x, const Ptensors2<float>& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",[](const Ptensors2<float>& x, const Ptensors2<float>& y){return cat_channels(x,y);})
  .def("cat_channels_back0",[](Ptensors2<float>& x, const Ptensors2<float>& r){return x.cat_channels_back0(r);})
  .def("cat_channels_back1",[](Ptensors2<float>& x, const Ptensors2<float>& r){return x.cat_channels_back1(r);})
//.def("cat_channels_back0",&Ptensors2<float>::cat_channels_back0)
//.def("cat_channels_back1",&Ptensors2<float>::cat_channels_back1)

  .def_static("cat",&Ptensors2<float>::cat)
  .def("add_cat_back",[](Ptensors2<float>& x, Ptensors2<float>& r, const int offs){
      x.get_grad()+=r.get_grad().slices(0,offs,x.dim(0));})

//.def("outer",&Ptensors2::outer)
//.def("outer_back0",&Ptensors2::outer_back0)
//.def("outer_back1",&Ptensors2::outer_back1)

  .def("scale_channels",[](Ptensors2<float>& x, at::Tensor& y){
      return scale_channels(x,ATview<float>(y));})
  .def("add_scale_channels_back0",[](Ptensors2<float>& r, const Ptensors2<float>& g, at::Tensor& y){
      r.add_scale_channels_back(g,ATview<float>(y));})

  .def("mprod",[](const Ptensors2<float>& x, at::Tensor& M){
      return mprod(x,ATview<float>(M));})
  .def("add_mprod_back0",[](Ptensors2<float>& r, const Ptensors2<float>& g, at::Tensor& M){
      r.add_mprod_back0(g,ATview<float>(M));})
  .def("mprod_back1",[](const Ptensors2<float>& x, const Ptensors2<float>& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("linear",[](const Ptensors2<float>& x, at::Tensor& y, at::Tensor& b){
      return linear(x,ATview<float>(y),ATview<float>(b));})
  .def("add_linear_back0",[](Ptensors2<float>& r, const Ptensors2<float>& g, at::Tensor& y){
      r.add_linear_back0(g,ATview<float>(y));})
  .def("linear_back1",[](const Ptensors2<float>& x, const Ptensors2<float>& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const Ptensors2<float>& x, Ptensors2<float>& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",[](const Ptensors2<float>& x, const float alpha){
      return ReLU(x,alpha);})
  .def("add_ReLU_back",[](Ptensors2<float>& x, const Ptensors2<float>& g, const float alpha){
      x.add_ReLU_back(g,alpha);})

  .def("inp",[](const Ptensors2<float>& x, const Ptensors2<float>& y){return x.inp(y);})
  .def("diff2",[](const Ptensors2<float>& x, const Ptensors2<float>& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const Ptensors0<float>& x){
      return Ptensors2<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors1<float>& x){
      return Ptensors2<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors2<float>& x){
      return Ptensors2<float>::linmaps(x);}) 

  .def_static("gather",[](const Ptensors0<float>& x, const AtomsPack& a){
      return Ptensors2<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1<float>& x, const AtomsPack& a){
      return Ptensors2<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors2<float>& x, const AtomsPack& a){
      return Ptensors2<float>::gather(x,a);}) 

  .def_static("gather",[](const Ptensors0<float>& x, const vector<vector<int> >& a){
      return Ptensors2<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1<float>& x, const vector<vector<int> >& a){
      return Ptensors2<float>::gather(x,a);}) 
  .def("add_gather_back_alt",[](Ptensors2<float>& x, Ptensors1<float>& g){
      x.add_gather_back(g);})
  .def_static("gather",[](const Ptensors2<float>& x, const vector<vector<int> >& a){
      return Ptensors2<float>::gather(x,a);}) 

  .def("add_linmaps_back",[](Ptensors2<float>& x, Ptensors0<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors2<float>& x, Ptensors1<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors2<float>& x, Ptensors2<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](Ptensors2<float>& x, Ptensors0<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](Ptensors2<float>& x, Ptensors1<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back_alt",[](Ptensors2<float>& x, Ptensors1<float>& g){
      x.add_gather_back(g);})
  .def("add_gather_back",[](Ptensors2<float>& x, Ptensors2<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors2<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors2<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors2<float>::repr);
//.def("linmaps0",[](const Ptensors2<float>& x){return linmaps0(x);})
//.def("linmaps1",[](const Ptensors2<float>& x){return linmaps1(x);})
//.def("linmaps2",[](const Ptensors2<float>& x){return linmaps2(x);})
//  .def("gather0",[](const Ptensors2<float>& x, const AtomsPack& a){return gather0(x,a);})
//  .def("gather1",[](const Ptensors2<float>& x, const AtomsPack& a){return gather1(x,a);})
//  .def("gather2",[](const Ptensors2<float>& x, const AtomsPack& a){return gather2(x,a);})
