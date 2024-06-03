typedef cnine::ATview<float> TVIEW;
typedef PtensorLayer<float> TLAYER;


pybind11::class_<PtensorLayer<float> >(m,"ptensorlayer")

  .def(py::init([](const int k, at::Tensor& M){
	return TLAYER(k,TVIEW(M));}))
  .def(py::init([](const int k, const AtomsPack& atoms, at::Tensor& M){
	return TLAYER(k,atoms,TVIEW(M));}))
  .def(py::init([](const int k, const vector<vector<int> >& atoms, at::Tensor& M){
	return TLAYER(k,AtomsPack(atoms),TVIEW(M));}))

  .def_static("create",[](const int k, const int n, const int _nc, const int fcode, const int _dev){
      return PtensorLayer<float>(k,AtomsPack(n),_nc,fcode,_dev);}, 
    py::arg("k"),py::arg("n"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const int k, const vector<vector<int> > _atoms, const int _nc, const int fcode, const int _dev){
      return PtensorLayer<float>(k,AtomsPack(_atoms),_nc,fcode,_dev);}, 
    py::arg("k"),py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def_static("create",[](const int k, const AtomsPack& _atoms, const int _nc, const int fcode, const int _dev){
      return PtensorLayer<float>(k,_atoms,cnine::channels=_nc,cnine::filltype=fcode,cnine::device=_dev);}, 
    py::arg("k"),py::arg("atoms"),py::arg("nc"),py::arg("fcode")=0,py::arg("device")=0)

  .def("copy",[](const PtensorLayer<float>& x){return x.copy();})
  .def("copy",[](const PtensorLayer<float>& x, const int _dev){return x.copy(_dev);})
  .def("zeros_like",&TLAYER::zeros_like)
  .def("randn_like",&TLAYER::gaussian_like)

  .def("torch",[](const TLAYER& x){return x.torch();})


// ---- Conversions, transport, etc. ------------------------------------------------------------------------


  .def("add_to_grad",[](PtensorLayer<float>& x, at::Tensor& y){
      x.add_to_grad(y);})
  .def("add_to_grad",[](PtensorLayer<float>& x, const PtensorLayer<float>& y, const float c){
      x.add_to_grad(y,c);})
  .def("get_grad",[](const PtensorLayer<float>& x){return x.get_grad();})
//.def("get_gradp",&PtensorLayer<float>::get_gradp)
//.def("gradp",&PtensorLayer<float>::get_gradp)

//.def("add_to_grad",[](PtensorLayer<float>& x, const int i, at::Tensor& T){
//x.get_grad().view_of_tensor(i).add(RtensorA::view(T));
//    })

//  .def("view_of_grad",&PtensorLayer<float>::view_of_grad)

  .def("__getitem__",[](const PtensorLayer<float>& x, const int i){return x(i);})
  .def("torch",[](const PtensorLayer<float>& x){return x.torch();})


// ---- Access ----------------------------------------------------------------------------------------------


  .def("get_dev",&PtensorLayer<float>::get_dev)
  .def("get_nc",&PtensorLayer<float>::get_nc)
  .def("get_atoms",[](const PtensorLayer<float>& x){return *x.atoms.obj->atoms;})
  .def("dim",&TLAYER::dim)
//.def("view_of_atoms",[](const PtensorLayer<float>& x){return x.atoms;})
//.def("view_of_atoms",&PtensorLayer<float>::view_of_atoms)


//.def("atoms_of",[](const PtensorLayer<float>& x, const int i){return vector<int>(x.atoms_of(i));})
   //  .def("push_back",&PtensorLayer<float>::push_back)

  .def("to_device",&PtensorLayer<float>::move_to_device)
   //.def("move_to_device_back",[](PtensorLayer<float>& x, const cnine::loose_ptr<PtensorLayer<float>>& g, const int dev){
   // if(!x.grad) x.grad=new PtensorLayer<float>(g,dev);
   // else x.grad->add(PtensorLayer<float>(g,dev));})


// ---- Operations -------------------------------------------------------------------------------------------

  
  .def("add",[](PtensorLayer<float>& r, const PtensorLayer<float>& x){r.add(x);})
  .def("add_back",[](PtensorLayer<float>& x, const PtensorLayer<float>& g){x.add_to_grad(g.get_grad());})

  .def("cat_channels",&TLAYER::cat_channels)
  .def("cat_channels_back0",&TLAYER::cat_channels_back0)
  .def("cat_channels_back1",&TLAYER::cat_channels_back1)

  .def_static("cat",&TLAYER::cat)
  .def("add_cat_back",[](TLAYER& x, TLAYER& r, const int offs){
      x.get_grad()+=r.slices(0,offs,x.dim(0));})

//.def("outer",&TLAYER::outer)
//.def("outer_back0",&TLAYER::outer_back0)
//.def("outer_back1",&TLAYER::outer_back1)

  .def("mprod",[](const PtensorLayer<float>& x, at::Tensor& M){
      return x.mprod(TVIEW(M));})
  .def("add_mprod_back0",[](PtensorLayer<float>& r, const PtensorLayer<float>& g, at::Tensor& M){
      r.add_mprod_back0(g,TVIEW(M));})
  .def("mprod_back1",[](const PtensorLayer<float>& x, const PtensorLayer<float>& g){
      return (x.transp()*g.get_grad()).torch();})

  .def("mult_channels",[](PtensorLayer<float>& x, at::Tensor& y){
      return x.mult_channels(TVIEW(y));})
  .def("add_mult_channels_back0",[](PtensorLayer<float>& r, const PtensorLayer<float>& g, at::Tensor& y){
      return r.add_mult_channels_back(g,TVIEW(y));})

  .def("linear",[](const PtensorLayer<float>& x, at::Tensor& y, at::Tensor& b){
      return linear(x,TVIEW(y),TVIEW(b));})
  .def("add_linear_back0",[](PtensorLayer<float>& r, const PtensorLayer<float>& g, at::Tensor& y){
      r.add_linear_back0(g,TVIEW(y));})
  .def("linear_back1",[](const PtensorLayer<float>& x, const PtensorLayer<float>& g){
      return (x.transp()*g.get_grad()).torch();})
  .def("linear_back2",[](const PtensorLayer<float>& x, PtensorLayer<float>& g){
      return g.get_grad().sum(0).torch();})

  .def("ReLU",&TLAYER::ReLU)
  .def("add_ReLU_back",&TLAYER::add_ReLU_back)

  .def("inp",[](const TLAYER& x, const TLAYER& y){return x.inp(y);})
  .def("diff2",[](const TLAYER& x, const TLAYER& y){return x.diff2(y);})

// ---- I/O --------------------------------------------------------------------------------------------------

  .def("str",&PtensorLayer<float>::str,py::arg("indent")="")
  .def("__str__",&PtensorLayer<float>::str,py::arg("indent")="")
  .def("__repr__",&PtensorLayer<float>::repr);




//.def("ReLU",[](const TLAYER& x, const float alpha){return x.ReLU(alpha);})
