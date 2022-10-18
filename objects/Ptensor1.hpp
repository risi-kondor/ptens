#ifndef _ptens_Ptensor1
#define _ptens_Ptensor1

#include "Ptens_base.hpp"
#include "Atoms.hpp"
#include "RtensorA.hpp"
#include "RtensorObj.hpp"
#include "Ptensor0.hpp"

#include "Ptensor1_xview.hpp"


namespace ptens{

  class Ptensor1: public cnine::RtensorA{
  public:

    int k;
    int nc;
    Atoms atoms;

    typedef cnine::RtensorA rtensor;
    typedef cnine::Gdims Gdims;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor1(const Atoms& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      rtensor(cnine::Gdims(_atoms.size(),_nc),dummy,_dev),
      atoms(_atoms),
      k(_atoms.size()), 
      nc(_nc){
    }


    // ---- Constructors -------------------------------------------------------------------------------------
    

    static Ptensor1 raw(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor1 zero(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor1 gaussian(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor1 gaussian(const Atoms& _atoms, const int nc, const float sigma, const int _dev){
      return Ptensor1(_atoms,nc,cnine::fill_gaussian(sigma),_dev);}

    static Ptensor1 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Copying ------------------------------------------------------------------------------------------


    Ptensor1(const Ptensor1& x):
      RtensorA(x), atoms(x.atoms){
      k=x.k;
      nc=x.nc;
    }

    Ptensor1(Ptensor1&& x):
      RtensorA(std::move(x)), atoms(std::move(x.atoms)){
      k=x.k;
      nc=x.nc;
    }

    Ptensor1& operator=(const Ptensor1& x)=delete;


    // ---- Conversions --------------------------------------------------------------------------------------


    Ptensor1(RtensorA&& x, Atoms&& _atoms):
      RtensorA(std::move(x)),
      atoms(std::move(_atoms)){
      assert(x.getk()==2);
      k=dims(0);
      nc=dims.back();
     }


    #ifdef _WITH_ATEN
    static Ptensor1 view(at::Tensor& x, Atoms&& _atoms){
      return Ptensor1(RtensorA::view(x),std::move(_atoms));
    }
    #endif 
 

    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return dims(0);
    }

    int get_nc() const{
      return dims.back();
    }

    float at_(const int i, const int c) const{
      return (*this)(atoms(i),c);
    }

    void inc_(const int i, const int c, float x){
      inc(atoms(i),c,x);
    }


    Rtensor2_view view() const{
      return view2();
    }

    Rtensor2_view view(const int offs, const int n) const{
      assert(offs+n<=nc);
      return view2().block(0,offs,k,n);
    }

    Ptensor1_xview view(const vector<int>& ix) const{
      return Ptensor1_xview(arr,nc,strides[0],strides[1],ix,dev);
    }

    Ptensor1_xview view(const vector<int>& ix, const int offs, const int n) const{
      return Ptensor1_xview(arr+strides[1]*offs,n,strides[0],strides[1],ix,dev);
    }


    // ---- Linmaps ------------------------------------------------------------------------------------------


    // 0 -> 1 
    void add_linmaps(const Ptensor0& x, int offs=0){ // 1
      PTENS_K_SAME(x);
      PTENS_CHANNELS(offs+1*x.nc<=nc);
      offs+=broadcast0(x.view1(),offs); // 1*1
    }
    
    void add_linmaps_back_to(Ptensor0& x, int offs=0) const{ // 1
      PTENS_K_SAME(x)
      view(offs,x.nc).sum0_into(x.view1());
    }

    
    // 1->1 
    void add_linmaps(const Ptensor1& x, int offs=0){ // 2 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+2*x.nc<=nc);
      offs+=broadcast0(x.reduce0().view1(),offs); // 1*1
      offs+=broadcast1(x.view(),offs); // 1*1
    }
    
    void add_linmaps_back(const Ptensor1& x, int offs=0){ // 2 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+2*nc<=x.nc);
      broadcast0(x.reduce0(offs,nc).view2());
      broadcast1(x.view(offs+nc,nc));
    }
    

    // 1 -> 0
    void add_linmaps_to(Ptensor0& x, int offs=0) const{ // 1 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+1*nc<=x.nc);
      offs+=x.broadcast0(reduce0(),offs); // 1*1
    }
    
    void add_linmaps_back(const Ptensor0& x, int offs=0){ // 1 
      PTENS_K_SAME(x)
      PTENS_CHANNELS(offs+1*nc<=x.nc);
      view()+=repeat0(x.view(offs,nc),k);
    }


  public: // ---- Reductions ---------------------------------------------------------------------------------


    rtensor reduce0() const{
      auto R=rtensor::zero(nc);
      view().sum0_into(R.view1());
      return R;
    }

    rtensor reduce0(const int offs, const int n) const{
      auto R=rtensor::zero(n);
      view(offs,n).sum0_into(R.view1());
      return R;
    }

    rtensor reduce0(const vector<int>& ix) const{
      auto R=rtensor::zero(Gdims(nc));
      view(ix).sum0_into(R.view1());
      return R;
    }

    rtensor reduce0(const vector<int>& ix, const int offs, const int n) const{
      auto R=rtensor::zero(Gdims(n));
      view(ix,offs,n).sum0_into(R.view1());
      return R;
    }


    rtensor reduce1() const{
      auto R=rtensor::zero({k,nc});
      R.view2().add(view());
      return R;
    }

    rtensor reduce1(const int offs, const int n) const{
      auto R=rtensor::zero({k,n});
      R.view2().add(view(offs,n));
      return R;
    }

    rtensor reduce1(const vector<int>& ix) const{
      auto R=rtensor::zero({(int)ix.size(),nc});
      R.view2().add(view(ix));
      return R;
    }

    rtensor reduce1(const vector<int>& ix, const int offs, const int n) const{
      auto R=rtensor::zero({(int)ix.size(),n});
      R.view2().add(view(ix,offs,n));
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const rtensor& x){
      view()+=repeat0(x.view1(),k);
    }

    int broadcast0(const rtensor& x, const int offs){
      view(offs,x.dim(0))+=repeat0(x.view1(),k);
      return x.dim(0);
    }

    void broadcast0(const rtensor& x, const vector<int>& ix){
      view(ix)+=repeat0(x.view1(),ix.size());
    }

    int broadcast0(const rtensor& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.dim(0))+=repeat0(x.view1(),ix.size());
      return x.dim(0);
    }


    void broadcast1(const rtensor& x){
      add(x.view2());
    }

    int broadcast1(const rtensor& x, const int offs){
      view(offs,x.dim(1))+=x.view2();
      return x.dim(1);
    }

    void broadcast1(const rtensor& x, const vector<int>& ix){
      view(ix)+=x.view2();
    }

    int broadcast1(const rtensor& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.dim(1))+=x.view2();
      return x.dim(1);
    }


  private: // ---- Broadcasting -------------------------------------------------------------------------------
    // These methods are deprectated / on hold 

    void broadcast0(const Rtensor1_view& x){
      view()+=repeat0(x,k);
    }

    int broadcast0(const Rtensor1_view& x, const int offs){
      view(offs,x.n0)+=repeat0(x,k);
      return x.n0;
    }

    void broadcast0(const Rtensor1_view& x, vector<int>& ix){
      view(ix)+=repeat0(x,ix.size());
    }

    int broadcast0(const Rtensor1_view& x, vector<int>& ix, const int offs){
      view(ix,offs,x.n0)+=repeat0(x,ix.size());
      return x.n0;
    }


    void broadcast1(const Rtensor2_view& x){
      add(x);
    }

    int broadcast1(const Rtensor2_view& x, const int offs){
      view(offs,x.n1)+=x;
      return x.n1;
    }

    void broadcast1(const Rtensor2_view& x, const vector<int>& ix){
      view(ix)+=x;
    }

    int broadcast1(const Rtensor2_view& x, const vector<int>& ix, const int offs){
      view(ix,offs,x.n1)+=x;
      return x.n1;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="")const{
      ostringstream oss;
      oss<<indent<<"Ptensor1 "<<atoms<<":"<<endl;
      oss<<view2().str(indent);
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor1& x){
      stream<<x.str(); return stream;}

  };

}


#endif 

/*
  namespace std{
    template<>
    struct hash<ptens::Ptensor1sgntr>{
    public:
      size_t operator()(const ptens::Ptensor1sgntr& sgntr) const{
	return (hash<int>()(sgntr.k)<<1)^hash<int>()(sgntr.nc); 
      }
    };
  }
*/




  /*
  class Ptensor1sgntr{
  public:

    int k;
    int nc;
    
    Ptensor1sgntr(const int _k, const int _nc): k(_k), nc(_nc){}

    bool operator==(const Ptensor1sgntr& x){
      return (k==x.k)&&(nc==x.nc);
    }
    
  };
  */

    /*
    rtensor reductions1(const vector<int>& ix, const int c) const{
      const int k=ix.size();
      rtensor R=rtensor::raw(cnine::Gdims(k));
      for(int i=0; i<k; i++)
	R.set(i,(*this)(ix[i],c));
      return R;
    }

    rtensor reductions0(const vector<int>& ix, const int c) const{
      const int n=dim(0);
      const int k=ix.size();
      rtensor R=rtensor::raw(cnine::Gdims(2));
      {float t=0; for(int i=0; i<k; i++) t+=(*this)(ix[i],c); R.set(0,t);}
      {float t=0; for(int i=0; i<n; i++) t+=(*this)(i,c); R.set(1,t);}
      return R;
    }
    */

    /*
    Ptensor1(const Ptensor1& x, const Atoms& _atoms):
      Ptensor1(_atoms,5*x.get_nc(),cnine::fill_zero()){
      pull_msg(x);
    }


    void pull_msg(const Ptensor1& x){
      int nc=x.get_nc();
      assert(get_nc()==5*nc);

      Atoms common=atoms.intersect(x.atoms);
      int k=common.size();
      vector<int> ix(atoms(common));
      vector<int> xix(x.atoms(common));

      for(int j=0; j<nc; j++){

	float s=0;
	for(int i=0; i<dims[0]; i++)
	  s+=x(i,j);
     
	float t=0;
	for(int i=0; i<k; i++){
	  t+=x(xix[i],j);
	}

	for(int i=0; i<k; i++){
	  inc(ix[i],5*j,x(xix[i],j));
	  inc(ix[i],5*j+1,t);
	  inc(ix[i],5*j+2,s);
	}

	for(int i=0; i<dims[0]; i++){
	  inc(i,5*j+3,t);
	  inc(i,5*j+4,s);
	}

      }

    }
    */
    /*
    void broadcast1(const rtensor& R1, const vector<int>& ix, int coffs){
      const int k=ix.size();
      //const int n=dims(0);

      for(int s=0; s<R1.dim(1); s++){
	for(int i=0; i<k; i++)
	  inc(ix[i],coffs,R1(i,s));
	coffs+=1;
      }
    }

    
   void broadcast0(const rtensor& R0, const vector<int>& ix, int coffs){
      const int k=ix.size();
      const int n=dims(0);

      for(int s=0; s<R0.dim(1); s++){
	for(int i=0; i<k; i++)
	  inc(ix[i],coffs,R0(i,s));
	for(int i=0; i<n; i++)
	  inc(i,coffs+1,R0(i,s));
	coffs+=2;
      }
    }
    */
