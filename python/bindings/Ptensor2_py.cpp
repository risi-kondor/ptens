pybind11::class_<Ptensor2<float>>(m,"ptensor2")

  .def(py::init([](const vector<int>& _atoms, const int _nc, const int _fcode, const int _dev){
	return Ptensor2<float>(_atoms,_nc,_fcode,_dev);}))

  .def_static("view",[](const vector<int>& v, at::Tensor& x){
      return Ptensor2<float>(v,tensorf::view(x));})

  .def("torch",[](const Ptensor2<float>& x){return x.torch();})


  .def("str",&Ptensor2<float>::str,py::arg("indent")="")
  .def("__str__",&Ptensor2<float>::str,py::arg("indent")="");


//  .def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
//      return Ptensor2<float>::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
//  .def_static("sequential",[](const Atoms& _atoms, const int _nc, const int _dev){
//      return Ptensor2<float>::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

//  .def_static("sequential",[](const vector<int> _atoms, const int _nc, const int _dev){
//      return Ptensor2<float>::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)


//   .def("add_linmaps",[](Ptensor2<float>& obj, const Ptensor0& x, int offs){
//       return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)
//   .def("add_linmaps",[](Ptensor2<float>& obj, const Ptensor1& x, int offs){
//       return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)
//   .def("add_linmaps",[](Ptensor2<float>& obj, const Ptensor2<float>& x, int offs){
//       return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)

//   .def("add_linmaps_to",[](const Ptensor2<float>& obj, Ptensor0& x, int offs){
//       return obj.add_linmaps_to(x,offs);}, py::arg("x"), py::arg("offs")=0)
//   .def("add_linmaps_to",[](const Ptensor2<float>& obj, Ptensor1& x, int offs){
//       return obj.add_linmaps_to(x,offs);}, py::arg("x"), py::arg("offs")=0)

