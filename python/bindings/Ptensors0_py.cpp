typedef cnine::ATview<float> TVIEW;

pybind11::class_<Ptensors0<float> >(m,"ptensors0")

  .def_static("view",[](const AtomsPack& atoms, at::Tensor& x){
      return Ptensors0<float>(atoms,tensorf::view(x));})

  .def("add_reduce0_to",[](Ptensors0f& obj, const Ptensors0f& r, const AindexPack& list){
      obj.add_reduce0_to(r,list);})
  
  .def("broadcast0",[](Ptensors0f& obj, const Ptensors0f& x, const AindexPack& list, const int offs){
      obj.broadcast0(x,list,offs);},py::arg("x"),py::arg("list"),py::arg("offs")=0)


  .def("str",&Ptensors0<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensors0<float>::str,py::arg("indent")="")
  .def("__repr__",&Ptensors0<float>::repr);



/*
  .def(py::init([](const AtomsPack& atoms, at::Tensor& M){
	return Ptensors0<float>(atoms,Ltensor<float>(M));}))
  .def(py::init([](const vector<vector<int> >& atoms, at::Tensor& M){
	return Ptensors0<float>(AtomsPack(atoms),Ltensor<float>(M));}))

  .def_static("create",[](const int n, const int _nc, const int fcode, const int _dev){
      return Ptensors0<float>(n,_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors0<float>(AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return Ptensors0<float>(_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)
*/

//.def("like",[](const Ptensors0<float>& x, at::Tensor& M){
//      return Ptensors0<float>(x.atoms,ATview<float>(M));})
 //  .def("copy",[](const Ptensors0<float>& x){return x.copy();})
//   .def("copy",[](const Ptensors0<float>& x, const int _dev){return x.copy(_dev);})
//   .def("zeros_like",[](const Ptensors0<float>& x){return Ptensors0<float>::zeros_like(x);})
//   .def("randn_like",[](const Ptensors0<float>& x){return Ptensors0<float>::gaussian_like(x);})



// ---- Conversions, transport, etc. ------------------------------------------------------------------------

/*
  .def("__len__",&Ptensors0<float>::size)
  .def("add_to_grad",[](Ptensors0<float>& x, at::Tensor& y){x.add_to_grad(ATview<float>(y));})
  .def("add_to_grad",[](Ptensors0<float>& x, const Ptensors0<float>& y, const float c){x.add_to_grad(y,c);})
  .def("add_to_grad",[](Ptensors0<float>& x, at::Tensor& y){x.get_grad().add(ATview<float>(y));})
  .def("add_to_grad",[](Ptensors0<float>& x, const Ptensors0<float>& y, const float c){x.get_grad().add(y,c);})
  .def("get_grad",[](Ptensors0<float>& x){return x.get_grad();})

  .def("__getitem__",[](const Ptensors0<float>& x, const int i){return x(i);})
  .def("torch",[](const Ptensors0<float>& x){return x.torch();})
*/

// ---- Access ----------------------------------------------------------------------------------------------


//   .def("get_dev",&Ptensors0<float>::get_dev)
//   .def("get_nc",&Ptensors0<float>::get_nc)
//   .def("get_atoms",[](const Ptensors0<float>& x){return x.get_atoms();})
//   .def("dim",&Ptensors0<float>::dim)

//   .def("to",[](const Ptensors0<float>& x, const int dev){return Ptensors0<float>(x,dev);})
//   .def("to_device",&Ptensors0<float>::move_to_device)


// ---- Operations -------------------------------------------------------------------------------------------


//   .def("add",[](Ptensors0<float>& r, const Ptensors0<float>& x){r.add(x);})
//   .def("add_back",[](Ptensors0<float>& x, const Ptensors0<float>& g){x.add_to_grad(g.get_grad());})

//   .def("cat_channels",[](const Ptensors0<float>& x, const Ptensors0<float>& y){return cat_channels(x,y);})
//   .def("cat_channels_back0",[](Ptensors0<float>& x, const Ptensors0<float>& r){return x.cat_channels_back0(r);})
//   .def("cat_channels_back1",[](Ptensors0<float>& x, const Ptensors0<float>& r){return x.cat_channels_back1(r);})
//  .def("cat_channels_back0",&Ptensors0<float>::cat_channels_back0)
//.def("cat_channels_back1",&Ptensors0<float>::cat_channels_back1)

//  .def("cat",&Ptensors0<float>::cat)
//  .def("add_cat_back",[](Ptensors0<float>& x, Ptensors0<float>& r, const int offs){
//      x.get_grad()+=r.get_grad().rows(offs,x.dim(0));})

//.def("outer",&Ptensors0::outer)
//.def("outer_back0",&Ptensors0::outer_back0)
//.def("outer_back1",&Ptensors0::outer_back1)

//   .def("scale_channels",[](Ptensors0<float>& x, at::Tensor& y){
//       return scale_channels(x,ATview<float>(y));})
//   .def("add_scale_channels_back0",[](Ptensors0<float>& r, const Ptensors0<float>& g, at::Tensor& y){
//       r.add_scale_channels_back(g,ATview<float>(y));})

//   .def("mprod",[](const Ptensors0<float>& x, at::Tensor& M){
//       return mprod(x,ATview<float>(M));})
//   .def("add_mprod_back0",[](Ptensors0<float>& r, const Ptensors0<float>& g, at::Tensor& M){
//       r.add_mprod_back0(g,ATview<float>(M));})
//   .def("mprod_back1",[](const Ptensors0<float>& x, const Ptensors0<float>& g){
//       return (x.transp()*g.get_grad()).torch();})

//   .def("linear",[](const Ptensors0<float>& x, at::Tensor& y, at::Tensor& b){
//       return linear(x,ATview<float>(y),ATview<float>(b));})
//   .def("add_linear_back0",[](Ptensors0<float>& r, const Ptensors0<float>& g, at::Tensor& y){
//       r.add_linear_back0(g,ATview<float>(y));})
//   .def("linear_back1",[](const Ptensors0<float>& x, const Ptensors0<float>& g){
//       return (x.transp()*g.get_grad()).torch();})
//   .def("linear_back2",[](const Ptensors0<float>& x, Ptensors0<float>& g){
//       return g.get_grad().sum(0).torch();})

//   .def("ReLU",[](const Ptensors0<float>& x, const float alpha){
//       return ReLU(x,alpha);})
//   .def("add_ReLU_back",[](Ptensors0<float>& x, const Ptensors0<float>& g, const float alpha){
//       x.add_ReLU_back(g,alpha);})

//   .def("inp",[](const Ptensors0<float>& x, const Ptensors0<float>& y){return x.inp(y);})
//   .def("diff2",[](const Ptensors0<float>& x, const Ptensors0<float>& y){return x.diff2(y);})


// ---- Message passing --------------------------------------------------------------------------------------


/*
  .def_static("linmaps",[](const Ptensors0<float>& x){
      return Ptensors0<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors1<float>& x){
      return Ptensors0<float>::linmaps(x);}) 
  .def_static("linmaps",[](const Ptensors2<float>& x){
      return Ptensors0<float>::linmaps(x);}) 

  .def_static("gather",[](const Ptensors0<float>& x, const AtomsPack& a){
      return Ptensors0<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1<float>& x, const AtomsPack& a){
      return Ptensors0<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors2<float>& x, const AtomsPack& a){
      return Ptensors1<float>::gather(x,a);}) 

  .def_static("gather",[](const Ptensors0<float>& x, const vector<vector<int> >& a){
      return Ptensors0<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors1<float>& x, const vector<vector<int> >& a){
      return Ptensors0<float>::gather(x,a);}) 
  .def_static("gather",[](const Ptensors2<float>& x, const vector<vector<int> >& a){
      return Ptensors0<float>::gather(x,a);}) 

  .def("add_linmaps_back",[](Ptensors0<float>& x, Ptensors0<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors0<float>& x, Ptensors1<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})
  .def("add_linmaps_back",[](Ptensors0<float>& x, Ptensors2<float>& g){
      x.get_grad().add_linmaps_back(g.get_grad());})

  .def("add_gather_back",[](Ptensors0<float>& x, Ptensors0<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back",[](Ptensors0<float>& x, Ptensors1<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
  .def("add_gather_back_alt",[](Ptensors0<float>& x, Ptensors1<float>& g){
      x.add_gather_back(g);})
  .def("add_gather_back",[](Ptensors0<float>& x, Ptensors2<float>& g){
      x.get_grad().add_gather_back(g.get_grad());})
*/

//  .def("add_reduce0",[](Ptensors0f& obj, const Ptensors1f& x, const int offs=0, const int nc=0){
//      obj.add_reduce0(x,offs,nc);})


// ---- I/O --------------------------------------------------------------------------------------------------




