
m.def("add_msg",[](Ptensor0& r, const Ptensor0& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor1& r, const Ptensor0& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor2& r, const Ptensor0& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_msg",[](Ptensor0& r, const Ptensor1& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor1& r, const Ptensor1& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor2& r, const Ptensor1& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_msg",[](Ptensor0& r, const Ptensor2& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor1& r, const Ptensor2& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_msg",[](Ptensor2& r, const Ptensor2& x, int offs){return add_msg(r,x,offs);}, 
  py::arg("r"), py::arg("x"), py::arg("offs")=0);



