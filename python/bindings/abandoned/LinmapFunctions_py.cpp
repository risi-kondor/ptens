
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


m.def("add_linmaps1to0",[](Ptensor0& r, const Ptensor1& x, int offs){
    return x.add_linmaps_to(r,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to0_back",[](Ptensor1& r, const Ptensor0& x, int offs){
    return r.add_linmaps_back(x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps1to1",[](Ptensor1& r, const Ptensor1& x, int offs){
    return r.add_linmaps(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to1_back",[](Ptensor1& r, const Ptensor1& x, int offs){
    return r.add_linmaps_back(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps1to2",[](Ptensor2& r, const Ptensor1& x, int offs){
    return r.add_linmaps(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to2_back",[](Ptensor1& r, const Ptensor2& x, int offs){
    return x.add_linmaps_back_to(r,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps2to0",[](Ptensor0& r, const Ptensor2& x, int offs){
    return x.add_linmaps_to(r,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to0_back",[](Ptensor2& r, const Ptensor0& x, int offs){
    return r.add_linmaps_back(x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps2to1",[](Ptensor1& r, const Ptensor2& x, int offs){
    return x.add_linmaps_to(r,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to1_back",[](Ptensor2& r, const Ptensor1& x, int offs){
    return r.add_linmaps_back(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps2to2",[](Ptensor2& r, const Ptensor2& x, int offs){
    return r.add_linmaps(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to2_back",[](Ptensor2& r, const Ptensor2& x, int offs){
    return r.add_linmaps_back(x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);


// -----------------------------------------------------------------------------------------------------------


/*
m.def("add_linmaps0to0",[](Ptensors0& r, const Ptensors0& x, int offs){
    return add_linmaps(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to0_back",[](Ptensors0& r, const Ptensors0& x, int offs){
    return add_linmaps_back(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
//m.def("add_linmaps0to0_back",[](Ptensors0* r, const Ptensors0* x, int offs){
//    return add_linmaps_back(*r,*x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to0_back",[](loose_ptr<Ptensors0>& r, const loose_ptr<Ptensors0>& x, int offs){
    return add_linmaps_back(*r,*x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps0to1",[](Ptensors1& r, const Ptensors0& x, int offs){
    return add_linmaps(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to1_back",[](Ptensors0& r, const Ptensors1& x, int offs){
    return add_linmaps_back(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
//m.def("add_linmaps0to1_back",[](Ptensors0* r, const Ptensors1* x, int offs){
//    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to1_back",[](loose_ptr<Ptensors0>& r, loose_ptr<Ptensors1>& x, int offs){
    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps0to2",[](Ptensors2& r, const Ptensors0& x, int offs){
    return add_linmaps(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to2_back",[](Ptensors0& r, const Ptensors2& x, int offs){
    return add_linmaps_back(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
//m.def("add_linmaps0to2_back",[](Ptensors0* r, const Ptensors2* x, int offs){
//    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps0to2_back",[](loose_ptr<Ptensors0>& r, const loose_ptr<Ptensors2>& x, int offs){
    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps1to0",[](Ptensors0& r, const Ptensors1& x, int offs){
    return add_linmaps(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to0_back",[](Ptensors1& r, const Ptensors0& x, int offs){
    return add_linmaps_back(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to0_back",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors0>& x, int offs){
    return add_linmaps_back(*r,*x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps1to0_n",[](Ptensors0& r, const Ptensors1& x, int offs){
    return add_linmaps_n(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to0_back_n",[](Ptensors1& r, const Ptensors0& x, int offs){
    return add_linmaps_back_n(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to0_back_n",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors0>& x, int offs){
    return add_linmaps_back_n(*r,*x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps1to1",[](Ptensors1& r, const Ptensors1& x, int offs){
    return add_linmaps(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to1_back",[](Ptensors1& r, const Ptensors1& x, int offs){
    return add_linmaps_back(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to1_back",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors1>& x, int offs){
    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps1to1_n",[](Ptensors1& r, const Ptensors1& x, int offs){
    return add_linmaps_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to1_back_n",[](Ptensors1& r, const Ptensors1& x, int offs){
    return add_linmaps_back_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to1_back_n",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors1>& x, int offs){
    return add_linmaps_back_n(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps1to2",[](Ptensors2& r, const Ptensors1& x, int offs){
    return add_linmaps(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to2_back",[](Ptensors1& r, const Ptensors2& x, int offs){
    return add_linmaps_back(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to2_back",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors2>& x, int offs){
    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps1to2_n",[](Ptensors2& r, const Ptensors1& x, int offs){
    return add_linmaps_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to2_back_n",[](Ptensors1& r, const Ptensors2& x, int offs){
    return add_linmaps_back_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps1to2_back_n",[](loose_ptr<Ptensors1>& r, const loose_ptr<Ptensors2>& x, int offs){
    return add_linmaps_back_n(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps2to0",[](Ptensors0& r, const Ptensors2& x, int offs){
    return add_linmaps(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to0_back",[](Ptensors2& r, const Ptensors0& x, int offs){
    return add_linmaps_back(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to0_back",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors0>& x, int offs){
    return add_linmaps_back(*r,*x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps2to0_n",[](Ptensors0& r, const Ptensors2& x, int offs){
    return add_linmaps_n(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to0_back_n",[](Ptensors2& r, const Ptensors0& x, int offs){
    return add_linmaps_back_n(r,x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to0_back_n",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors0>& x, int offs){
    return add_linmaps_back_n(*r,*x,offs);}, py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps2to1",[](Ptensors1& r, const Ptensors2& x, int offs){
    return add_linmaps(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to1_back",[](Ptensors2& r, const Ptensors1& x, int offs){
    return add_linmaps_back(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to1_back",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors1>& x, int offs){
    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps2to1_n",[](Ptensors1& r, const Ptensors2& x, int offs){
    return add_linmaps_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to1_back_n",[](Ptensors2& r, const Ptensors1& x, int offs){
    return add_linmaps_back_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to1_back_n",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors1>& x, int offs){
    return add_linmaps_back_n(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);


m.def("add_linmaps2to2",[](Ptensors2& r, const Ptensors2& x, int offs){
    return add_linmaps(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to2_back",[](Ptensors2& r, const Ptensors2& x, int offs){
    return add_linmaps_back(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to2_back",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors2>& x, int offs){
    return add_linmaps_back(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);

m.def("add_linmaps2to2_n",[](Ptensors2& r, const Ptensors2& x, int offs){
    return add_linmaps_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to2_back_n",[](Ptensors2& r, const Ptensors2& x, int offs){
    return add_linmaps_back_n(r,x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
m.def("add_linmaps2to2_back_n",[](loose_ptr<Ptensors2>& r, const loose_ptr<Ptensors2>& x, int offs){
    return add_linmaps_back_n(*r,*x,offs);},  py::arg("r"), py::arg("x"), py::arg("offs")=0);
*/
