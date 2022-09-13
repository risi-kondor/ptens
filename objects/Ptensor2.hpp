#ifndef _ptens_Ptensor2
#define _ptens_Ptensor2

#include "Ptens_base.hpp"
#include "Atoms.hpp"
#include "RtensorObj.hpp"
#include "Ptensor1.hpp"
#include "Ptensor2_xview.hpp"

namespace ptens{

  class Ptensor2: public cnine::RtensorA{
  public:

    int k;
    int nc;
    Atoms atoms;

    typedef cnine::Gdims Gdims;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor2(const Atoms& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      rtensor(cnine::Gdims(_atoms.size(),_atoms.size(),_nc),dummy,_dev),
      atoms(_atoms), k(_atoms.size()), nc(_nc){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor2 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor2 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor2 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor2 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor2(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Copying ------------------------------------------------------------------------------------------


    Ptensor2(const Ptensor2& x):
      RtensorA(x), atoms(x.atoms){
      k=x.k;
      nc=x.nc;
    }

    Ptensor2(Ptensor2&& x):
      RtensorA(std::move(x)), atoms(std::move(x.atoms)){
      k=x.k;
      nc=x.nc;
    }

    Ptensor2& operator=(const Ptensor2& x)=delete;


    // ---- Conversions --------------------------------------------------------------------------------------


    Ptensor2(RtensorA&& x, Atoms&& _atoms):
      RtensorA(std::move(x)),
      atoms(std::move(_atoms)){
      assert(x.getk()==3);
      k=dims(0);
      nc=dims.back();
    }
 

    #ifdef _WITH_ATEN
    static Ptensor2 view(at::Tensor& x, Atoms&& _atoms){
      return Ptensor2(RtensorA::view(x),std::move(_atoms));
    }
    #endif 


    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return k;
    }

    int get_nc() const{
      return nc;
      //      return dims.back();
    }

    float at_(const int i, const int j, const int c) const{
      return (*this)(atoms(i),atoms(j),c);
    }

    void inc_(const int i, const int j, const int c, float x){
      inc(atoms(i),atoms(j),c,x);
    }


    Rtensor3_view view() const{
      return view3();
    }

    Rtensor3_view view(const int offs, const int n) const{
      assert(offs+n<=nc);
      return view3().block(0,0,offs,k,k,n);
    }
    
    Ptensor2_xview view(const vector<int>& ix) const{
      return Ptensor2_xview(arr,nc,strides[0],strides[1],strides[2],ix,dev);
    }

    Ptensor2_xview view(const vector<int>& ix, const int offs, const int n) const{
      return Ptensor2_xview(arr+strides[2]*offs,n,strides[0],strides[1],strides[2],ix,dev);
    }


    // ---- Linmaps ------------------------------------------------------------------------------------------


    // 0 -> 2
    void add_linmaps(const Ptensor0& x, int offs=0){ // 2
      assert(offs+2*x.nc<=nc);
      offs+=broadcast(x.view1(),offs); // 2*1
    }

    void add_linmaps_back_to(Ptensor0& x, int offs=0) const{ // 2
      assert(offs+2*x.nc<=nc);
      x.add(reduce0(offs,x.nc));
    }


    // 1 -> 2
    void add_linmaps(const Ptensor1& x, int offs=0){ // 5 
      assert(x.k==k);
      assert(offs+5*x.nc<=nc);
      offs+=broadcast(x.reduce0().view1(),offs); // 2*1
      offs+=broadcast(x.view2(),offs); // 3*1
    }

    void add_linmaps_back_to(Ptensor1& x, int offs=0) const{ // 5 
      assert(x.k==k);
      assert(offs+5*x.nc<=nc);
      x.broadcast(reduce0(offs,x.nc).view1());
      x.broadcast(reduce1(offs+2*x.nc,x.nc).view1());
    }
    

    // 2 -> 2
    void add_linmaps(const Ptensor2& x, int offs=0){ // 15
      assert(x.k==k);
      assert(offs+15*x.nc<=nc);
      offs+=broadcast(x.reduce0().view1(),offs); // 2*2
      offs+=broadcast(x.reduce1().view2(),offs); // 3*3
      offs+=broadcast(x.view3(),offs); // 2
    }
    
    void add_linmaps_back(const Ptensor2& x, int offs=0){ // 15 check offsets!!!
      assert(x.k==k);
      assert(offs+15*nc<=x.nc);
      broadcast(x.reduce0(offs,nc).view1()); // 2*2
      broadcast(x.reduce1(offs+2*nc,nc).view2()); // 3*3
      broadcast(x.view(offs+5*nc,nc)); // 2 
    }
    

    // 2 -> 0 
    void add_linmaps_to(Ptensor0& x, int offs=0) const{ // 2
      assert(offs+2*nc<=x.nc);
      offs+=x.broadcast(reduce0().view1(),offs); // 1*2
    }
    
    void add_linmaps_back(const Ptensor0& x, int offs=0){ // 2
      assert(offs+2*nc<=x.nc);
      //offs+=x.broadcast(reduce0().view1(),offs); // 1*2
    }
    

    // 2 -> 1
    void add_linmaps_to(Ptensor1& x, int offs=0) const{ // 5 
      assert(x.k==k);
      assert(offs+5*nc<=x.nc);
      offs+=x.broadcast(reduce0().view1(),offs); // 1*2
      offs+=x.broadcast(reduce1().view2(),offs); // 1*3
    }
    
    void add_linmaps_back(const Ptensor1& x, int offs=0){ // 5 
      assert(x.k==k);
      assert(offs+5*nc<=x.nc);
      //offs+=x.broadcast(reduce0().view1(),offs); // 1*2
      //offs+=x.broadcast(reduce1().view2(),offs); // 1*3
    }
    

  public: // ---- Reductions ---------------------------------------------------------------------------------


    rtensor reduce0() const{ // 2
      auto R=rtensor::zero({k,2*nc});
      view().sum01_into(R.view1().block(0,nc));
      view().diag01().sum0_into(R.view1().block(nc,nc));
      return R;
    }

    rtensor reduce0(const int offs, const int n) const{ // 2
      auto R=rtensor::zero({k,n});
      view(offs,n).sum01_into(R.view1());
      view(offs+n,n).diag01().sum0_into(R.view1());
      return R;
    }

    rtensor reduce0(const vector<int>& ix) const{
      auto R=rtensor::zero(Gdims(2*nc));
      view(ix).sum01_into(R.view1().block(0,nc));
      view(ix).diag01().sum0_into(R.view1().block(nc,nc));
      return R;
    }

    rtensor reduce0(const vector<int>& ix, const int offs, const int n) const{
      auto R=rtensor::zero(Gdims(n));
      view(ix,offs,n).sum01_into(R.view1());
      view(ix,offs+n,n).diag01().sum0_into(R.view1());
      return R;
    }


    rtensor reduce1() const{
      auto R=rtensor::zero({k,3*nc});
      view().sum0_into(R.view2().block(0,0,k,nc));
      view().sum1_into(R.view2().block(0,nc,k,nc));
      R.view2().block(0,2*nc,k,nc)+=view().diag01();
      return R;
    }

    rtensor reduce1(const int offs, const int n) const{
      auto R=rtensor::zero({k,n});
      view(offs,n).sum0_into(R.view2());
      view(offs+n,n).sum1_into(R.view2());
      R.view()+=view(offs+2*n,n).diag01();
      return R;
    }

    rtensor reduce1(const vector<int>& ix) const{
      auto R=rtensor::zero({k,3*nc});
      view(ix).sum0_into(R.view2().block(0,0,k,nc));
      view(ix).sum1_into(R.view2().block(0,nc,k,nc));
      R.view2().block(0,2*nc,k,nc).add(view(ix).diag01());
      return R;
    }

    rtensor reduce1(const vector<int>& ix, const int offs, const int n) const{
      auto R=rtensor::zero({k,n});
      view(ix,offs,n).sum0_into(R.view2());
      view(ix,offs+n,n).sum1_into(R.view2());
      R.view2().add(view(ix,offs+2*n,n).diag01());
      return R;
    }


    rtensor reduce2() const{
      auto R=rtensor::zero({k,k,nc});
      R.view3().add(view());
      return R;
    }

    rtensor reduce2(const int offs, const int n) const{ // flipping
      auto R=rtensor::zero({k,k,2*n});
      R.view3().block(0,0,0,k,k,n).add(view(offs,n));
      R.view3().block(0,0,n,k,k,2*n).add(view(offs,n).transp01());
      return R;
    }

    rtensor reduce2(const vector<int>& ix) const{
      auto R=rtensor::zero({(int)ix.size(),(int)ix.size(),nc});
      R.view3().add(view(ix));
      return R;
    }

    rtensor reduce2(const vector<int>& ix, const int offs, const int n) const{ // flipping
      int K=ix.size();
      auto R=rtensor::zero({(int)ix.size(),(int)ix.size(),2*n});
      R.view3().block(0,0,0,K,K,n).add(view(ix,offs,n));
      R.view3().block(0,0,0,K,K,n).add(view(ix,offs,n).transp01());
      return R;
    }

    
  public: // ---- Broadcasting -------------------------------------------------------------------------------


    int broadcast(const Rtensor1_view& x, const int offs){
      int n=x.n0;
      assert(2*n+offs<=nc);
      view(offs,n)+=repeat0(repeat0(x,k),k);
      view(offs+n,n).diag01()+=repeat0(x,k);
      return 2*n;
    }

    void broadcast(const Rtensor1_view& x){
      int n=x.n0;
      assert(n==2*nc);
      view()+=repeat0(repeat0(x.block(0,n),k),k);
      view().diag01()+=repeat0(x.block(n,n),k);
    }

    int broadcast(const vector<int>& ix, const Rtensor1_view& x, const int offs){
      int n=x.n0;
      int K=ix.size();
      assert(2*n+offs<=nc);
      view(ix,offs,n)+=repeat0(repeat0(x,K),K);
      view(ix,offs+n,n).diag01()+=repeat0(x,K);
      return 2*n;
    }

    void broadcast(const vector<int>& ix, const Rtensor1_view& x){
      int n=x.n0;
      int K=ix.size();
      assert(n==2*nc);
      view(ix)+=repeat0(repeat0(x.block(0,n),K),K);
      view(ix).diag01()+=repeat0(x.block(0,n),K);
    }


    int broadcast(const Rtensor2_view& x, const int offs){
      int n=x.n1;
      assert(3*n+offs<=nc);
      view(offs,n)+=repeat0(x,k);
      view(offs+n,n)+=repeat1(x,k);
      view(offs+2*n,n).diag01()+=x;
      return 3*n;
    }

    void broadcast(const Rtensor2_view& x){
      int n=x.n1;
      assert(n==3*nc);
      view()+=repeat0(x.block(0,0,k,nc),k);
      view()+=repeat1(x.block(0,nc,k,nc),k);
      view().diag01()+=x.block(0,2*nc,k,nc);
    }

    int broadcast(const vector<int>& ix, const Rtensor2_view& x, const int offs){
      int n=x.n1;
      int K=ix.size();
      assert(3*n+offs<=nc);
      view(ix,offs,n)+=repeat0(x,K);
      view(ix,offs+n,n)+=repeat1(x,K);
      view(ix,offs+2*n,n).diag01()+=x;
      return 3*n;
    }

    void broadcast(const vector<int>& ix, const Rtensor2_view& x){
      int n=x.n1;
      int K=ix.size();
      assert(n==3*nc);
      view(ix)+=repeat0(x.block(0,0,K,nc),K);
      view(ix)+=repeat1(x.block(0,nc,K,nc),K);
      view(ix).diag01()+=x.block(0,2*nc,K,nc);
    }

  
    int broadcast(const Rtensor3_view& x, const int offs){
      int n=x.n2;
      assert(2*n+offs<=nc);
      view(offs,n)+=x;
      view(offs+n,n)+=x.transp01();
      return 2*n;
    }

    void broadcast(const Rtensor3_view& x){ // no flipping
      int n=x.n2;
      assert(n==nc);
      view()+=x;
    }

    int broadcast(const vector<int>& ix, const Rtensor3_view& x, const int offs){
      int n=x.n2;
      assert(2*n+offs<=nc);
      view(ix,offs,n)+=x;
      view(ix,offs+n,n)+=x.transp01();
      return 2*n;
    }

    void broadcast(const vector<int>& ix, const Rtensor3_view& x){ // no flipping
      int n=x.n2;
      assert(n==nc);
      view(ix)+=x;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Ptensor2"<<atoms<<":"<<endl;
      for(int c=0; c<get_nc(); c++){
	oss<<indent<<"channel "<<c<<":"<<endl;
	oss<<view3().slice2(c).str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const Ptensor2& x){
      stream<<x.str(); return stream;}

  };


}


#endif 
    /*
    rtensor reductions0e(const rtensor& R1, const vector<int>& ix, const int c) const{
      const int k=ix.size();
      const int n=dims(0);
      rtensor R=rtensor::zero({5});

      for(int i=0; i<k; i++){
	  R.inc(0,R1(i,0));
	  R.inc(2,R1(i,2));
	  R.inc(3,R1(i,3));
	  R.inc(4,R1(i,4));
	}

      float t=0;
      for(int i=0; i<n; i++)
	for(int j=0; j<n; j++)
	  t+=(*this)(i,j,c);
      R.inc(1,t);
      
      return R;
    }


    rtensor reductions1e(const vector<int>& ix, const int c) const{
      const int k=ix.size();
      const int n=dims(0);
      rtensor R=rtensor::raw({k,5});
      
      for(int i=0; i<k; i++){
	int _i=ix[i];
	{float s=0; for(int j=0; j<k; j++) s+=(*this)(_i,ix[j],c); R.set(i,0,s);}
	{float s=0; for(int j=0; j<k; j++) s+=(*this)(ix[j],_i,c); R.set(i,1,s);}
	{float s=0; for(int j=0; j<n; j++) s+=(*this)(_i,j,c); R.set(i,2,s);}
	{float s=0; for(int j=0; j<n; j++) s+=(*this)(j,_i,c); R.set(i,3,s);}
	R.set(i,4,(*this)(_i,_i,c));
      }

      return R;
    }
    */
    /*
    Ptensor2(const Ptensor2& x, const Atoms& _atoms):
      Ptensor2(_atoms,52*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor2(const Ptensor1& x, const Atoms& _atoms):
      Ptensor2(_atoms,15*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }

    Ptensor1 msg1(const Atoms& _atoms){
      Ptensor1 R(_atoms,15*get_nc(),cnine::fill_zero());
      push_msg(R);
      return R;
    }


    void pull_msg(const Ptensor1& x){
      int nc=x.get_nc();
      const int cstride=15;
      assert(get_nc()==cstride*nc);

      Atoms common=atoms.intersect(x.atoms);
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int c=0; c<nc; c++){
	rtensor R1=x.reductions1(xix,c);
	broadcast1(R1,ix,cstride*c);
	rtensor R0=x.reductions0(xix,c);
	broadcast0(R0,ix,cstride*c+5);
      }
      
    }


    void pull_msg(const Ptensor2& x){
      int nc=x.get_nc();
      const int cstride=52;
      assert(get_nc()==cstride*nc);

      Atoms common=atoms.intersect(x.atoms);
      int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));


      for(int c=0; c<nc; c++){

	for(int i=0; i<k; i++)
	  for(int j=0; j<k; j++){
	    inc(ix[i],ix[j],cstride*c,x(xix[i],xix[j],c));
	    inc(ix[j],ix[i],cstride*c+1,x(xix[i],xix[j],c));
	}

	rtensor R1=x.reductions1e(xix,c);
	broadcast1(R1,ix,cstride*c+2);

	rtensor R0=x.reductions0e(R1,xix,c);
	broadcast0(R0,ix,cstride*c+27);

      }
      
    }


    void push_msg(Ptensor1& x) const{
      int nc=get_nc();
      const int cstride=15;
      assert(x.get_nc()==cstride*nc);

      Atoms common=atoms.intersect(x.atoms);
      //int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int c=0; c<nc; c++){
	rtensor R1=reductions1e(ix,c);
	x.broadcast1(R1,xix,cstride*c);
	rtensor R0=reductions0e(R1,ix,c);
	x.broadcast0(R0,xix,cstride*c+5);
      }
      
    }
    */
      //for(int c=0; c<nc; c++){
      //auto slice=view3().slice2(c);
      //slice.sum0_into(R.view2().slice1(c));
      //slice.sum1_into(R.view2().slice1(c+nc));
      //R.view2().slice1(c+2*nc)=slice.diag();
      //}
      //for(int c=0; c<n; c++){
      //auto slice=view().slice2(c+offs);
      //R.set(c,view().slice2(c+offs).sum()+view().slice2(c+n+offs).diag().sum());
      //}
      //for(int c=0; c<nc; c++){
      //auto slice=view3().slice2(c);
      //R.set(c,slice.sum());
      //R.set(nc+c,slice.diag().sum());
      //}
    /*
    void broadcast1(const rtensor& R1, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++)
	    inc(_i,ix[j],coffs,R1(i,s));
	  for(int j=0; j<k; j++)
	    inc(ix[j],_i,coffs+1,R1(i,s));
	  for(int j=0; j<n; j++)
	    inc(_i,j,coffs+2,R1(i,s));
	  for(int j=0; j<n; j++)
	    inc(j,_i,coffs+3,R1(i,s));
	  inc(_i,_i,coffs+4,R1(i,s));
	}
	coffs+=5;
      }
    }
    

    void broadcast0(const rtensor& R0, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R0.dim(0); s++){
	float v=R0(s);

	for(int i=0; i<k; i++){
	  int _i=ix[i];
	  for(int j=0; j<k; j++)
	    inc(_i,ix[j],coffs,v);
	  for(int j=0; j<n; j++){
	    inc(_i,j,coffs+1,v);
	    inc(j,_i,coffs+2,v);
	  }
	  inc(_i,_i,coffs+3,v);
	}

	for(int i=0; i<n; i++)
	  for(int j=0; j<n; j++)
	    inc(i,j,coffs+4,v);
	coffs+=5;

      }      
    }
    */

