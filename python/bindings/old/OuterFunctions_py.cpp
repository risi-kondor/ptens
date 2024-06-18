
m.def("outer",[](const Ptensors0& x, const Ptensors0& y){return outer(x,y);});
m.def("add_outer",[](Ptensors0& r, const Ptensors0& x, const Ptensors0& y){return add_outer(r,x,y);});
m.def("add_outer_back0",[](Ptensors0& xg, const Ptensors0& g, const Ptensors0& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](Ptensors0& yg, const Ptensors0& g, const Ptensors0& x){return add_outer_back1(yg,g,x);});
m.def("add_outer_back0",[](loose_ptr<Ptensors0>& xg, const loose_ptr<Ptensors0>& g, const Ptensors0& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](loose_ptr<Ptensors0>& yg, const loose_ptr<Ptensors0>& g, const Ptensors0& x){return add_outer_back1(yg,g,x);});

m.def("outer",[](const Ptensors0& x, const Ptensors1& y){return outer(x,y);});
m.def("add_outer",[](Ptensors1& r, const Ptensors0& x, const Ptensors1& y){return add_outer(r,x,y);});
m.def("add_outer_back0",[](Ptensors0& xg, const Ptensors1& g, const Ptensors1& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](Ptensors1& yg, const Ptensors1& g, const Ptensors0& x){return add_outer_back1(yg,g,x);});
m.def("add_outer_back0",[](loose_ptr<Ptensors0>& xg, const loose_ptr<Ptensors1>& g, const Ptensors1& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](loose_ptr<Ptensors1>& yg, const loose_ptr<Ptensors1>& g, const Ptensors0& x){return add_outer_back1(yg,g,x);});

m.def("outer",[](const Ptensors1& x, const Ptensors0& y){return outer(x,y);});
m.def("add_outer",[](Ptensors1& r, const Ptensors1& x, const Ptensors0& y){return add_outer(r,x,y);});
m.def("add_outer_back0",[](Ptensors1& xg, const Ptensors1& g, const Ptensors0& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](Ptensors0& yg, const Ptensors1& g, const Ptensors1& x){return add_outer_back1(yg,g,x);});
m.def("add_outer_back0",[](loose_ptr<Ptensors1>& xg, const loose_ptr<Ptensors1>& g, const Ptensors0& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](loose_ptr<Ptensors0>& yg, const loose_ptr<Ptensors1>& g, const Ptensors1& x){return add_outer_back1(yg,g,x);});

m.def("outer",[](const Ptensors1& x, const Ptensors1& y){return outer(x,y);});
m.def("add_outer",[](Ptensors2& r, const Ptensors1& x, const Ptensors1& y){return add_outer(r,x,y);});
m.def("add_outer_back0",[](Ptensors1& xg, const Ptensors2& g, const Ptensors1& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](Ptensors1& yg, const Ptensors2& g, const Ptensors1& x){return add_outer_back1(yg,g,x);});
m.def("add_outer_back0",[](loose_ptr<Ptensors1>& xg, const loose_ptr<Ptensors2>& g, const Ptensors1& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](loose_ptr<Ptensors1>& yg, const loose_ptr<Ptensors2>& g, const Ptensors1& x){return add_outer_back1(yg,g,x);});

m.def("outer",[](const Ptensors0& x, const Ptensors2& y){return outer(x,y);});
m.def("add_outer",[](Ptensors2& r, const Ptensors0& x, const Ptensors2& y){return add_outer(r,x,y);});
m.def("add_outer_back0",[](Ptensors0& xg, const Ptensors2& g, const Ptensors2& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](Ptensors2& yg, const Ptensors2& g, const Ptensors0& x){return add_outer_back1(yg,g,x);});
m.def("add_outer_back0",[](loose_ptr<Ptensors0>& xg, const loose_ptr<Ptensors2>& g, const Ptensors2& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](loose_ptr<Ptensors2>& yg, const loose_ptr<Ptensors2>& g, const Ptensors0& x){return add_outer_back1(yg,g,x);});

m.def("outer",[](const Ptensors2& x, const Ptensors0& y){return outer(x,y);});
m.def("add_outer",[](Ptensors2& r, const Ptensors2& x, const Ptensors0& y){return add_outer(r,x,y);});
m.def("add_outer_back0",[](Ptensors2& xg, const Ptensors2& g, const Ptensors0& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](Ptensors0& yg, const Ptensors2& g, const Ptensors2& x){return add_outer_back1(yg,g,x);});
m.def("add_outer_back0",[](loose_ptr<Ptensors2>& xg, const loose_ptr<Ptensors2>& g, const Ptensors0& y){return add_outer_back0(xg,g,y);});
m.def("add_outer_back1",[](loose_ptr<Ptensors0>& yg, const loose_ptr<Ptensors2>& g, const Ptensors2& x){return add_outer_back1(yg,g,x);});

