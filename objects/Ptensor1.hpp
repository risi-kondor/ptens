#ifndef _ptens_Ptensor1
#define _ptens_Ptensor1

#include "Atoms.hpp"
#include "RtensorA.hpp"
#include "RtensorObj.hpp"
//#include "PtensorSgntr.hpp"


namespace ptens{

  class Ptensor1: public cnine::RtensorA{
  public:

    int k;
    int nc;
    Atoms atoms;

    typedef cnine::RtensorA rtensor;
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

    static Ptensor1 sequential(const Atoms& _atoms, const int nc, const int _dev=0){
      return Ptensor1(_atoms,nc,cnine::fill_sequential(),_dev);}

    
    // ---- Conversions --------------------------------------------------------------------------------------


    Ptensor1(Atoms&& _atoms, RtensorA&& x):
      RtensorA(std::move(x)),
      atoms(std::move(_atoms)){}
 

    // ---- Access -------------------------------------------------------------------------------------------


    int getk() const{
      return dims(0);
    }

    int get_nc() const{
      return dims.back();
    }

    //PtensorSgntr signature() const{
    //return PtensorSgntr(getk(),get_nc());
    //}

    float at_(const int i, const int c) const{
      return (*this)(atoms(i),c);
    }

    void inc_(const int i, const int c, float x){
      inc(atoms(i),c,x);
    }


    // ---- Linmaps ------------------------------------------------------------------------------------------


    void add_linmaps(const Ptensor0& x, int offs=0){ // 1
      assert(offs+1*x.nc<=nc);
      offs+=broadcast(x.view1(),offs); // 1*1
    }
    
    void add_linmaps(const Ptensor1& x, int offs=0){ // 2 
      assert(x.k==k);
      assert(offs+2*x.nc<=nc);
      offs+=broadcast(x.reductions0().view1(),offs); // 1*1
      offs+=broadcast(x.view2(),offs); // 1*1
    }
    
    void add_linmaps_to(Ptensor0& x, int offs=0) const{ // 1 
      assert(offs+1*nc<=x.nc);
      offs+=x.broadcast(reductions0().view1(),offs); // 1*1
    }
    

    Ptensor0 reductions0() const{ // 1
      auto R=Ptensor0::raw(atoms,nc);
      view2().sum0_into(R.view1());
      return R;
    }

    int broadcast(const Rtensor1_view& x, const int offs=0){ // 1
      int n=x.n0;
      assert(n+offs<=nc);
      view2().block(0,offs,k,n)+=repeat0(x,k);
      return n;
    }

    int broadcast(const Rtensor2_view& x, const int offs=0){ // 1
      int n=x.n1;
      assert(x.n0==k);
      assert(n+offs<=nc);
      view2().block(0,offs,k,n)+=x;
      return n;
    }


    // ---- Message passing ----------------------------------------------------------------------------------


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


  public: // ---- Reductions ---------------------------------------------------------------------------------


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


  public: // ---- Broadcasting -------------------------------------------------------------------------------


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
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="")const{
      ostringstream oss;
      oss<<indent<<"Ptensor1"<<atoms<<":"<<endl;
      oss<<view2().transp().str(indent);
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

