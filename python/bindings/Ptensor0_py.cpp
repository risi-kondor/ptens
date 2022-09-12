pybind11::class_<Ptensor0>(m,"ptensor0")

  .def_static("raw",[](const Atoms& _atoms, const int _nc, const int _dev){
      return Ptensor0::raw(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)
  .def_static("sequential",[](const Atoms& _atoms, const int _nc, const int _dev){
      return Ptensor0::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("sequential",[](const vector<int> _atoms, const int _nc, const int _dev){
      return Ptensor0::sequential(_atoms,_nc,_dev);}, py::arg("atoms"),py::arg("nc"),py::arg("device")=0)

  .def_static("view",[](at::Tensor& x, const vector<int>& v){return Ptensor0::view(x,v);})
  .def("torch",[](const Ptensor0& x){return x.torch();})

  .def("atomsv",&Ptensor0::atomsv)

  .def("add_linmaps",[](Ptensor0& obj, const Ptensor0& x, int offs){
      return obj.add_linmaps(x,offs);}, py::arg("x"), py::arg("offs")=0)

  .def("str",&Ptensor0::str,py::arg("indent")="")
  .def("__str__",&Ptensor0::str,py::arg("indent")="");


m.def("add_linmaps0to0",[](Ptensor0& r, const Ptensor0& x, int offs){
    return r.add_linmaps(x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to0_back",[](Ptensor0& r, const Ptensor0& x, int offs){
    return r.add_linmaps_back(x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps0to1",[](Ptensor1& r, const Ptensor0& x, int offs){
    return r.add_linmaps(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to1_back",[](Ptensor0& r, const Ptensor1& x, int offs){
    return x.add_linmaps_back_to(r,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps0to2",[](Ptensor2& r, const Ptensor0& x, int offs){
    return r.add_linmaps(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to2_back",[](Ptensor0& r, const Ptensor2& x, int offs){
    return x.add_linmaps_back_to(r,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

  
