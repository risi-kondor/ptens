pybind11::class_<Ptensor2>(m,"ptensor2")

  .def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
      return Ptensor2::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const Atoms& _atoms, const int _nc, const int _dev){
      return Ptensor2::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("sequential",[](const vector<int> _atoms, const int _nc, const int _dev){
      return Ptensor2::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("view",[](at::Tensor& x, const vector<int>& v){return Ptensor2::view(x,v);})
  .def("torch",[](const Ptensor2& x){return x.torch();})

  .def("add_linmaps",[](Ptensor2& obj, const Ptensor0& x, int offs){
      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)
  .def("add_linmaps",[](Ptensor2& obj, const Ptensor1& x, int offs){
      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)
  .def("add_linmaps",[](Ptensor2& obj, const Ptensor2& x, int offs){
      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)

  .def("add_linmaps_to",[](const Ptensor2& obj, Ptensor0& x, int offs){
      return obj.add_linmaps_to(x,offs);}, py::arg("x"), py::arg("offs")=0)
  .def("add_linmaps_to",[](const Ptensor2& obj, Ptensor1& x, int offs){
      return obj.add_linmaps_to(x,offs);}, py::arg("x"), py::arg("offs")=0)

  .def("str",&Ptensor2::str,py::arg("indent")="")
  .def("__str__",&Ptensor2::str,py::arg("indent")="");


