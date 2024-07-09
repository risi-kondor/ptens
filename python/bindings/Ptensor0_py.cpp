pybind11::class_<Ptensor0<float> >(m,"ptensor0")

  .def(py::init([](const vector<int>& _atoms, const int _nc, const int _fcode, const int _dev){
	return Ptensor0<float>(_atoms,_nc,_fcode,_dev);}))

  .def_static("view",[](const vector<int>& v, at::Tensor& x){
      return Ptensor0<float>(v,tensorf::view(x));})

//.def_static("view",[](const AtomsPack& atoms, at::Tensor& x){
//    return Ptensor0<float>(atoms,tensorf(x));})

  .def("torch",[](const Ptensor0<float>& x){
      return x.torch();})

  .def("str",&Ptensor0<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensor0<float>::str,py::arg("indent")="");



  
//  .def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
//      return Ptensor0<float>::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//.def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
//      return Ptensor0<float>::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
//  .def_static("sequential",[](const Atoms& _atoms, const int _nc, const int _dev){
//    return Ptensor0<float>::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
//
//.def_static("sequential",[](const vector<int> _atoms, const int _nc, const int _dev){
//      return Ptensor0<float>::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def("atomsv",&Ptensor0<float>::atomsv)

//  .def("add_linmaps",[](Ptensor0<float>& obj, const Ptensor0<float>& x, int offs){
//      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)

