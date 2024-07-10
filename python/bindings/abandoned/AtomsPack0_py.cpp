pybind11::class_<AtomsPack0>(m,"pilot0")

  .def(py::init([](const AtomsPack& x){return AtomsPack0(x);}))

  .def("mmap",[](const AtomsPack0& x, const MessageList& list, const AtomsPack0& y){
      return x.message_map(list,y);})
  .def("mmap",[](const AtomsPack0& x, const MessageList& list, const AtomsPack1& y){
      return x.message_map(list,y);})
  .def("mmap",[](const AtomsPack0& x, const MessageList& list, const AtomsPack2& y){
      return x.message_map(list,y);})

  .def("str",[](const AtomsPack0& x){return x.str();})
  .def("__str__",[](const AtomsPack0& x){return x.str();});


