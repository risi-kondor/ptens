pybind11::class_<Ptensor1<float> >(m,"ptensor1")

  .def(py::init([](const vector<int>& _atoms, const int _nc, const int _fcode, const int _dev){
	return Ptensor1<float>(_atoms,_nc,_fcode,_dev);}))

  .def_static("view",[](const vector<int>& v, at::Tensor& x){
      return Ptensor1<float>(v,tensorf::view(x));})

  .def("torch",[](const Ptensor1<float>& x){
      return x.torch();})

  .def("str",&Ptensor1<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensor1<float>::str,py::arg("indent")="");




//   .def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
//       return Ptensor1<float>::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
//   .def_static("sequential",[](const Atoms& _atoms, const int _nc, const int _dev){
//       return Ptensor1<float>::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
//   .def_static("sequential",[](const vector<int> _atoms, const int _nc, const int _dev){
//       return Ptensor1<float>::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def("add_linmaps",[](Ptensor1<float>& obj, const Ptensor0<float>& x, int offs){
//      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)
//  .def("add_linmaps",[](Ptensor1<float>& obj, const Ptensor1<float>& x, int offs){
//      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)

//.def("add_linmaps_to",[](const Ptensor1<float>& obj, Ptensor0<float>& x, int offs){
//    return obj.add_linmaps_to(x,offs);}, py::arg("x"), py::arg("offs")=0)

