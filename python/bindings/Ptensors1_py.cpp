typedef cnine::ATview<float> TVIEW;

pybind11::class_<Ptensors1<float> >(m,"ptensors1")

  .def(py::init([](const AtomsPack& atoms, at::Tensor& M){
	return Ptensors1<float>(atoms,Ltensor<float>(M));}))
  .def(py::init([](const vector<vector<int> >& atoms, at::Tensor& M){
	return Ptensors1<float>(AtomsPack(atoms),Ltensor<float>(M));}))

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors1<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors1<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

//  .def("like",[](const Ptensors1<float>& x, at::Tensor& M){
//      return Ptensors1<float>(x.atoms,ATview<float>(M));})
  .def("copy",[](const Ptensors1<float>& x){return x.copy();})
  .def("copy",[](const Ptensors1<float>& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",[](const Ptensors1<float>& x){return Ptensors1<float>::zeros_like(x);})
  .def("randn_like",[](const Ptensors1<float>& x){return Ptensors1<float>::gaussian_like(x);})

  .def_static("view",[](const AtomsPack& atoms, at::Tensor& x){
      return Ptensors1<float>(atoms,Ltensor<float>::view(x));})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


//.def("add_to_grad",[](Ptensors1<float>& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
//.def("add_to_grad",[](Ptensors1<float>& x, const Ptensors1<float>& y, const float c){x.add_to_grad(y,c);})
//   .def("add_to_grad",[](Ptensors1<float>& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
//   .def("add_to_grad",[](Ptensors1<float>& x, const Ptensors1<float>& y, const float c){x.get_grad().add(y,c);})
//   .def("get_grad",[](Ptensors1<float>& x){return Ptensors1<float>(x.get_grad());})

//   .def("__getitem__",[](const Ptensors1<float>& x, const int i){return x(i);})
//   .def("torch",[](const Ptensors1<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("__len__",&Ptensors1<float>::size)
  .def("get_dev",&Ptensors1<float>::get_dev)
  .def("get_nc",&Ptensors1<float>::get_nc)
  .def("get_atoms",[](const Ptensors1<float>& x){return x.get_atoms();})
  .def("dim",&Ptensors1<float>::dim)

  .def("to",[](const Ptensors1<float>& x, const int dev){return Ptensors1<float>(x,dev);})
  .def("to_device",&Ptensors1<float>::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


//   .def("add",[](Ptensors1<float>& r, const Ptensors1<float>& x){r.add(x);})
//   .def("add_back",[](Ptensors1<float>& x, const Ptensors1<float>& g){x.add_to_grad(g.get_grad());})

//   .def("cat_channels",[](const Ptensors1<float>& x, const Ptensors1<float>& y){return cat_channels(x,y);})
//   .def("cat_channels_back0",[](Ptensors1<float>& x, const Ptensors1<float>& r){return x.cat_channels_back0(r);})
//   .def("cat_channels_back1",[](Ptensors1<float>& x, const Ptensors1<float>& r){return x.cat_channels_back1(r);})

  .def_static("cat",&Ptensors1<float>::cat)
  .def("add_cat_back",[](Ptensors1<float>& x, Ptensors1<float>& r, const int offs){
      x.get_grad()+=r.get_grad().slices(0,offs,x.dim(0));})

//.def("outer",&Ptensors1::outer)
//.def("outer_back0",&Ptensors1::outer_back0)
//.def("outer_back1",&Ptensors1::outer_back1)

//   .def("scale_channels",[](Ptensors1<float>& x, at::Tensor& y){
//       return scale_channels(x,ATview<float>(y));})
//   .def("add_scale_channels_back0",[](Ptensors1<float>& r, const Ptensors1<float>& g, at::Tensor& y){
//       r.add_scale_channels_back(g,ATview<float>(y));})

//   .def("mprod",[](const Ptensors1<float>& x, at::Tensor& M){
//       return mprod(x,ATview<float>(M));})
//   .def("add_mprod_back0",[](Ptensors1<float>& r, const Ptensors1<float>& g, at::Tensor& M){
//       r.add_mprod_back0(g,ATview<float>(M));})
//   .def("mprod_back1",[](const Ptensors1<float>& x, const Ptensors1<float>& g){
//       return (x.transp()*g.get_grad()).torch();})

//   .def("linear",[](const Ptensors1<float>& x, at::Tensor& y, at::Tensor& b){
//       return linear(x,ATview<float>(y),ATview<float>(b));})
//   .def("add_linear_back0",[](Ptensors1<float>& r, const Ptensors1<float>& g, at::Tensor& y){
//       cnine::fnlog timer("Ptensorsb::add_linear_back0()");
//       r.add_linear_back0(g,ATview<float>(y));})
//   .def("linear_back1",[](const Ptensors1<float>& x, const Ptensors1<float>& g){
//       cnine::fnlog timer("Ptensorsb::add_linear_back1()");
//       return (x.transp()*g.get_grad()).torch();})
//   .def("linear_back2",[](const Ptensors1<float>& x, Ptensors1<float>& g){
//       cnine::fnlog timer("Ptensorsb::add_linear_back2()");
//       return g.get_grad().sum(0).torch();})

//   .def("ReLU",[](const Ptensors1<float>& x, const float alpha){
//       return ReLU(x,alpha);})
//   .def("add_ReLU_back",[](Ptensors1<float>& x, const Ptensors1<float>& g, const float alpha){
//       x.add_ReLU_back(g,alpha);})

//   .def("inp",[](const Ptensors1<float>& x, const Ptensors1<float>& y){return x.inp(y);})
//   .def("diff2",[](const Ptensors1<float>& x, const Ptensors1<float>& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


  .def_static("linmaps",[](const Ptensors0<float>& x){
      return Ptensors1<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors1<float>& x){
      return Ptensors1<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors2<float>& x){
      return Ptensors1<float>::linmaps(x);}) 

  .def_static("gather",[](const Ptensors0<float>& x, const AtomsPack& a, const int min_overlaps){
      return Ptensors1<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1<float>& x, const AtomsPack& a, const int min_overlaps){
      return Ptensors1<float>::gather(x,a,min_overlaps);}) 
  .def_static("gather",[](const Ptensors2<float>& x, const AtomsPack& a, const int min_overlaps){
      return Ptensors1<float>::gather(x,a);}) 

  .def_static("gather",[](const Ptensors0<float>& x, const vector<vector<int> >& a, const int min_overlaps){
      return Ptensors1<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1<float>& x, const vector<vector<int> >& a, const int min_overlaps){
      return Ptensors1<float>::gather(x,a,min_overlaps);}) 
  .def_static("gather",[](const Ptensors2<float>& x, const vector<vector<int> >& a, const int min_overlaps){
      return Ptensors1<float>::gather(x,a);}) 

  .def("add_linmaps_back",[](Ptensors1<float>& x, Ptensors0<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors1<float>& x, Ptensors1<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors1<float>& x, Ptensors2<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](Ptensors1<float>& x, Ptensors0<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](Ptensors1<float>& x, Ptensors1<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back_alt",[](Ptensors1<float>& x, Ptensors1<float>& g){
      x.add_gather_back(g);})
  .def("add_gather_back",[](Ptensors1<float>& x, Ptensors2<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})

  .def("add_reduce0_to",[](const Ptensors1f& obj, const Ptensors0f& r, const int offs, const int nc){
      obj.add_reduce0_to(r,offs,nc);},py::arg("r"),py::arg("offs")=0,py::arg("nc")=0)


// ---- I/O --------------------------------------------------------------------------------------------------


  .def("str",&Ptensors1<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors1<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors1<float>::repr);



//  .def("linmaps0",[](const Ptensors1<float>& x){return linmaps0(x);})
//  .def("linmaps1",[](const Ptensors1<float>& x){return linmaps1(x);})
//  .def("linmaps2",[](const Ptensors1<float>& x){return linmaps2(x);})

//  .def("gather0",[](const Ptensors1<float>& x, const AtomsPack& a){return gather0(x,a);})
//  .def("gather1",[](const Ptensors1<float>& x, const AtomsPack& a){return gather1(x,a);})
//  .def("gather2",[](const Ptensors1<float>& x, const AtomsPack& a){return gather2(x,a);})
