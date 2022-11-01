#ifndef _ptens_OuterLayers
#define _ptens_OuterLayers

#include "Ptensor0.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2.hpp"
#include "Ptensors0.hpp"
#include "Ptensors1.hpp"
#include "Ptensors2.hpp"


namespace ptens{

  #ifdef _WITH_CUDA
  extern void Ptensors1_outer10_cu(cnine::RtensorPack& r, const cnine::RtensorPack& x, const cnine::RtensorPack& y, cudaStream_t& stream);
  #endif

  // ---- 0,0 -> 0 


  void add_outer(Ptensors0& r, const Ptensors0& x, const Ptensors0& y){
    CNINE_CPUONLY1(r);
    using namespace cnine;
    int xc=x.nc;
    int yc=y.nc;
    PTENS_ASSRT(r.nc==xc*yc);
    if(r.dev==0){
      r.for_each_view(x,y,[&](const Rtensor1_view& r, const Rtensor1_view& x, const Rtensor1_view& y){
	  for(int i=0; i<xc; i++)
	    for(int j=0; j<yc; j++)
	      r.inc(i*yc+j,x(i)*y(j));
	});
    }
  }


  void add_outer_back0(Ptensors0& xg, const Ptensors0& g, const Ptensors0& y){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=xg.nc;
    int yc=y.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      xg.for_each_view(g,y,[&](const Rtensor1_view& xg, const Rtensor1_view& g, const Rtensor1_view& y){
	  for(int i=0; i<xc; i++)
	    for(int j=0; j<yc; j++)
	      xg.inc(i,g(i*yc+j)*y(j));
	});
    }
  }


  void add_outer_back1(Ptensors0& yg, const Ptensors0& g, const Ptensors0& x){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=x.nc;
    int yc=yg.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      yg.for_each_view(g,x,[&](const Rtensor1_view& yg, const Rtensor1_view& g, const Rtensor1_view& x){
	  for(int i=0; i<xc; i++)
	    for(int j=0; j<yc; j++)
	      yg.inc(j,g(i*yc+j)*x(i));
	});
    }
  }


  // ---- 0,1 -> 1 


  void add_outer(Ptensors1& r, const Ptensors0& x, const Ptensors1& y){
    CNINE_CPUONLY1(r);
    using namespace cnine;
    int xc=x.nc;
    int yc=y.nc;
    PTENS_ASSRT(r.nc==xc*yc);
    if(r.dev==0){
      r.for_each_view(x,y,[&](const Rtensor2_view& r, const Rtensor1_view& x, const Rtensor2_view& y){
	  int k=r.n0;
	  PTENS_ASSRT(k==y.n0);
	  for(int a=0; a<k; a++)
	    for(int i=0; i<xc; i++)
	      for(int j=0; j<yc; j++)
		r.inc(a,i*yc+j,x(i)*y(a,j));
	});
    }
  }

  void add_outer_back0(Ptensors0& xg, const Ptensors1& g, const Ptensors1& y){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=xg.nc;
    int yc=y.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      xg.for_each_view(g,y,[&](const Rtensor1_view& xg, const Rtensor2_view& g, const Rtensor2_view& y){
	  int k=g.n0;
	  PTENS_ASSRT(k==y.n0);
	  for(int a=0; a<k; a++)
	    for(int i=0; i<xc; i++)
	      for(int j=0; j<yc; j++)
		xg.inc(i,g(a,i*yc+j)*y(a,j));
	});
    }
  }

  void add_outer_back1(Ptensors1& yg, const Ptensors1& g, const Ptensors0& x){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=x.nc;
    int yc=yg.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      yg.for_each_view(g,x,[&](const Rtensor2_view& yg, const Rtensor2_view& g, const Rtensor1_view& x){
	  int k=g.n0;
	  PTENS_ASSRT(k==yg.n0);
	  for(int a=0; a<k; a++)
	    for(int i=0; i<xc; i++)
	      for(int j=0; j<yc; j++)
		yg.inc(a,j,g(a,i*yc+j)*x(i));
	});
    }
  }


  // ---- 1,0 -> 1 


  void add_outer(Ptensors1& r, const Ptensors1& x, const Ptensors0& y){
    //CNINE_CPUONLY1(r);
    using namespace cnine;
    int xc=x.nc;
    int yc=y.nc;
    PTENS_ASSRT(r.nc==xc*yc);
    if(r.dev==0){
      r.for_each_view(x,y,[&](const Rtensor2_view& r, const Rtensor2_view& x, const Rtensor1_view& y){
	  int k=r.n0;
	  PTENS_ASSRT(k==x.n0);
	  for(int a=0; a<k; a++)
	    for(int i=0; i<xc; i++)
	      for(int j=0; j<yc; j++)
		r.inc(a,i*yc+j,x(a,i)*y(j));
	});
    }
    GPUCODE(CUDA_STREAM(r,x,y,stream));
  }


  void add_outer_back0(Ptensors1& xg, const Ptensors1& g, const Ptensors0& y){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=xg.nc;
    int yc=y.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      xg.for_each_view(g,y,[&](const Rtensor2_view& xg, const Rtensor2_view& g, const Rtensor1_view& y){
	  int k=g.n0;
	  PTENS_ASSRT(k==xg.n0);
	  for(int a=0; a<k; a++)
	    for(int i=0; i<xc; i++)
	      for(int j=0; j<yc; j++)
		xg.inc(a,i,g(a,i*yc+j)*y(j));
	});
    }
  }


  void add_outer_back1(Ptensors0& yg, const Ptensors1& g, const Ptensors1& x){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=x.nc;
    int yc=yg.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      yg.for_each_view(g,x,[&](const Rtensor1_view& yg, const Rtensor2_view& g, const Rtensor2_view& x){
	  int k=g.n0;
	  PTENS_ASSRT(k==x.n0);
	  for(int a=0; a<k; a++)
	    for(int i=0; i<xc; i++)
	      for(int j=0; j<yc; j++)
		yg.inc(j,g(a,i*yc+j)*x(a,i));
	});
    }
  }


  // ---- 1,1 -> 2 


  void add_outer(Ptensors2& r, const Ptensors1& x, const Ptensors1& y){
    CNINE_CPUONLY1(r);
    using namespace cnine;
    int xc=x.nc;
    int yc=y.nc;
    PTENS_ASSRT(r.nc==xc*yc);
    if(r.dev==0){
      r.for_each_view(x,y,[&](const Rtensor3_view& r, const Rtensor2_view& x, const Rtensor2_view& y){
	  int k=r.n0;
	  PTENS_ASSRT(k==x.n0);
	  PTENS_ASSRT(k==y.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  r.inc(a,b,i*yc+j,x(a,i)*y(b,j));
	});
    }
  }

  void add_outer_back0(Ptensors1& xg, const Ptensors2& g, const Ptensors1& y){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=xg.nc;
    int yc=y.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      xg.for_each_view(g,y,[&](const Rtensor2_view& xg, const Rtensor3_view& g, const Rtensor2_view& y){
	  int k=g.n0;
	  PTENS_ASSRT(k==xg.n0);
	  PTENS_ASSRT(k==y.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  xg.inc(a,i,g(a,b,i*yc+j)*y(b,j));
	});
    }
  }

  void add_outer_back1(Ptensors1& yg, const Ptensors2& g, const Ptensors1& x){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=x.nc;
    int yc=yg.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      yg.for_each_view(g,x,[&](const Rtensor2_view& yg, const Rtensor3_view& g, const Rtensor2_view& x){
	  int k=g.n0;
	  PTENS_ASSRT(k==x.n0);
	  PTENS_ASSRT(k==yg.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  yg.inc(b,j,g(a,b,i*yc+j)*x(a,i));
	});
    }
  }


  // ---- 0,2 -> 2 


  void add_outer(Ptensors2& r, const Ptensors0& x, const Ptensors2& y){
    CNINE_CPUONLY1(r);
    using namespace cnine;
    int xc=x.nc;
    int yc=y.nc;
    PTENS_ASSRT(r.nc==xc*yc);
    if(r.dev==0){
      r.for_each_view(x,y,[&](const Rtensor3_view& r, const Rtensor1_view& x, const Rtensor3_view& y){
	  int k=r.n0;
	  PTENS_ASSRT(k==y.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  r.inc(a,b,i*yc+j,x(i)*y(a,b,j));
	});
    }
  }


  void add_outer_back0(Ptensors0& xg, const Ptensors2& g, const Ptensors2& y){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=xg.nc;
    int yc=y.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      xg.for_each_view(g,y,[&](const Rtensor1_view& xg, const Rtensor3_view& g, const Rtensor3_view& y){
	  int k=g.n0;
	  PTENS_ASSRT(k==y.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  xg.inc(i,g(a,b,i*yc+j)*y(a,b,j));
	});
    }
  }

  void add_outer_back1(Ptensors2& yg, const Ptensors2& g, const Ptensors0& x){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=x.nc;
    int yc=yg.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      yg.for_each_view(g,x,[&](const Rtensor3_view& yg, const Rtensor3_view& g, const Rtensor1_view& x){
	  int k=g.n0;
	  PTENS_ASSRT(k==yg.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  yg.inc(a,b,j,g(a,b,i*yc+j)*x(i));
	});
    }
  }


  // ---- 2,0 -> 2 


  void add_outer(Ptensors2& r, const Ptensors2& x, const Ptensors0& y){
    CNINE_CPUONLY1(r);
    using namespace cnine;
    int xc=x.nc;
    int yc=y.nc;
    PTENS_ASSRT(r.nc==xc*yc);
    if(r.dev==0){
      r.for_each_view(x,y,[&](const Rtensor3_view& r, const Rtensor3_view& x, const Rtensor1_view& y){
	  int k=r.n0;
	  PTENS_ASSRT(k==x.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  r.inc(a,b,i*yc+j,x(a,b,i)*y(j));
	});
    }
  }

  void add_outer_back0(Ptensors2& xg, const Ptensors2& g, const Ptensors0& y){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=xg.nc;
    int yc=y.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      xg.for_each_view(g,y,[&](const Rtensor3_view& xg, const Rtensor3_view& g, const Rtensor1_view& y){
	  int k=g.n0;
	  PTENS_ASSRT(k==xg.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  xg.inc(a,b,i,g(a,b,i*yc+j)*y(j));
	});
    }
  }

  void add_outer_back1(Ptensors0& yg, const Ptensors2& g, const Ptensors2& x){
    CNINE_CPUONLY1(g);
    using namespace cnine;
    int xc=x.nc;
    int yc=yg.nc;
    PTENS_ASSRT(g.nc==xc*yc);
    if(g.dev==0){
      yg.for_each_view(g,x,[&](const Rtensor1_view& yg, const Rtensor3_view& g, const Rtensor3_view& x){
	  int k=g.n0;
	  PTENS_ASSRT(k==x.n0);
	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++)
	      for(int i=0; i<xc; i++)
		for(int j=0; j<yc; j++)
		  yg.inc(j,g(a,b,i*yc+j)*x(a,b,i));
	});
    }
  }


  // --------------------------------------------------------------------------------------------------------


  Ptensors0 outer(const Ptensors0& x, const Ptensors0& y){
    CNINE_DEVICE_EQ(x,y);
    Ptensors0 R=Ptensors0::zero(x.atoms,x.nc*y.nc,x.dev);
    add_outer(R,x,y);
    return R;
  }

  Ptensors1 outer(const Ptensors0& x, const Ptensors1& y){
    CNINE_DEVICE_EQ(x,y);
    Ptensors1 R=Ptensors1::zero(y.atoms,x.nc*y.nc,x.dev);
    add_outer(R,x,y);
    return R;
  }

  Ptensors1 outer(const Ptensors1& x, const Ptensors0& y){
    CNINE_DEVICE_EQ(x,y);
    Ptensors1 R=Ptensors1::zero(x.atoms,x.nc*y.nc,x.dev);
    add_outer(R,x,y);
    return R;
  }

  Ptensors2 outer(const Ptensors1& x, const Ptensors1& y){
    CNINE_DEVICE_EQ(x,y);
    PTENS_ASSRT(x.atoms==y.atoms);
    Ptensors2 R=Ptensors2::zero(x.atoms,x.nc*y.nc,x.dev);
    add_outer(R,x,y);
    return R;
  }

  Ptensors2 outer(const Ptensors0& x, const Ptensors2& y){
    CNINE_DEVICE_EQ(x,y);
    Ptensors2 R=Ptensors2::zero(y.atoms,x.nc*y.nc,x.dev);
    add_outer(R,x,y);
    return R;
  }

  Ptensors2 outer(const Ptensors2& x, const Ptensors0& y){
    CNINE_DEVICE_EQ(x,y);
    Ptensors2 R=Ptensors2::zero(x.atoms,x.nc*y.nc,x.dev);
    add_outer(R,x,y);
    return R;
  }



}

#endif 
