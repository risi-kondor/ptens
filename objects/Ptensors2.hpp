#ifndef _ptens_Ptensors2
#define _ptens_Ptensors2

#include "Rtensor3_view.hpp"
#include "Cgraph.hpp"
#include "RtensorPack.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor2.hpp"
#include "diff_class.hpp"


namespace ptens{


  class Ptensors2: public cnine::RtensorPack, public cnine::diff_class<Ptensors2>{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::RtensorPack RtensorPack;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc;
    AtomsPack atoms;
    bool is_view=false;


    ~Ptensors2(){
#ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors2(){}

    Ptensors2(const int _nc, const int _dev=0):
      RtensorPack(3,_dev), nc(_nc){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors2(const int _n, const int _k, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPack(_n,{_k,_k,_nc},dummy,_dev), atoms(_n,_k), nc(_nc){}


  public: // ----- Named constructors ------------------------------------------------------------------------


    static Ptensors2 raw(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors2(_n,_k,_nc,cnine::fill_raw(),_dev);}

    static Ptensors2 zero(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors2(_n,_k,_nc,cnine::fill_zero(),_dev);}

    static Ptensors2 sequential(const int _n, const int _k, const int _nc, const int _dev=0){
      Ptensors2 R(_n,_k,_nc,cnine::fill_raw(),_dev);
      for(int i=0; i<_n; i++)
	R.view3_of(i).set(i); 
      return R;
    }


    static Ptensors2 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors2 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor2::raw(_atoms(i),_nc));
      return R;
    }

    static Ptensors2 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors2 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor2::zero(_atoms(i),_nc));
      return R;
    }

    static Ptensors2 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors2 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor2::gaussian(_atoms(i),_nc));
      return R;
    }

    static Ptensors2 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors2 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++)
	R.push_back(Ptensor2::sequential(_atoms(i),_nc));
      return R;
    }


    static Ptensors2 concat(const Ptensors2& x, const Ptensors2& y){
      Ptensors2 R=Ptensors2::zero(x.atoms,x.nc+y.nc);
      R.add_to_channels(x,0);
      R.add_to_channels(y,x.nc);
      return R;
    }


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors2(const Ptensors2& x):
      RtensorPack(x),
      cnine::diff_class<Ptensors2>(x),
      atoms(x.atoms),
      nc(x.nc){
      PTENS_COPY_WARNING();
    }
	
    Ptensors2(Ptensors2&& x):
      RtensorPack(std::move(x)),
      cnine::diff_class<Ptensors2>(std::move(x)),
      atoms(std::move(x.atoms)),
      nc(x.nc){
      PTENS_COPY_WARNING();
    }

    Ptensors2& operator=(const Ptensors2& x)=delete;


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static Ptensors2* new_zeros_like(const Ptensors2& x){
      return new Ptensors2(RtensorPack::zeros_like(x),x.atoms,x.nc);
    }

    
  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors2(RtensorPack&& x, const AtomsPack& _atoms, const int _nc):
      RtensorPack(std::move(x)), atoms(_atoms), nc(_nc){}


  public: // ---- Transport ----------------------------------------------------------------------------------


    Ptensors2(const Ptensors2& x, const int _dev):
      RtensorPack(x,_dev),
      atoms(x.atoms),
      nc(x.nc){}

    Ptensors2& to_device(const int _dev){
      RtensorPack::to_device(_dev);
      return *this;
    }


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }

    AtomsPack view_of_atoms(){
      return atoms.view();
    }


    int k_of(const int i) const{
      return dim_of(i,0);
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }

    rtensor tensor_of(const int i) const{
      return RtensorPack::operator()(i);
    }

    Rtensor3_view view_of(const int i) const{
      return RtensorPack::view3_of(i);
    }

    Rtensor2_view fused_view_of(const int i) const{
      return RtensorPack::view3_of(i).fuse01();
    }

    Rtensor3_view view_of(const int i, const int offs, const int n) const{
      return RtensorPack::view3_of(i).block(0,0,offs,-1,-1,n);
    }

    Ptensor2_xview view_of(const int i, const vector<int>& ix) const{
      vector<int> v=headers(i);
      PTENS_ASSRT(v.size()==4);
      if(dev==1) return Ptensor2_xview(arrg+v[0],v[3],v[2]*v[3],v[3],1,ix,1);
      return Ptensor2_xview(arr+v[0],v[3],v[2]*v[3],v[3],1,ix,0);
    }

    Ptensor2_xview view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      vector<int> v=headers(i);
      PTENS_ASSRT(v.size()==4);
      if(dev==1) return Ptensor2_xview(arrg+v[0]+offs,n,v[2]*v[3],v[3],1,ix,1);
      return Ptensor2_xview(arr+v[0]+offs,n,v[2]*v[3],v[3],1,ix,0);
    }

    Ptensor2 operator()(const int i) const{
      return Ptensor2(tensor_of(i),atoms_of(i));
    }

    int push_back(const Ptensor2& x){
      if(size()==0) nc=x.get_nc();
      else PTENS_ASSRT(nc==x.get_nc());
      RtensorPack::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_to_channels(const Ptensors2& x, const int offs){
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i,offs,x.nc)+=x.view_of(i);
    }

    void add_channels(const Ptensors2& x, const int offs){
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i)+=x.view_of(i,offs,x.nc);
    }

    void add_mprod(const Ptensors2& x, const rtensor& y){
      PTENS_CPUONLY();
      PTENS_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	fused_view_of(i).add_matmul_AA(x.fused_view_of(i),y.view2());
    }

    void add_mprod_back0(const Ptensors2& g, const rtensor& y){
      PTENS_CPUONLY();
      PTENS_ASSRT(g.size()==size());
      for(int i=0; i<size(); i++)
	fused_view_of(i).add_matmul_AT(g.fused_view_of(i),y.view2());
    }

    void add_mprod_back1_to(rtensor& r, const Ptensors2& x) const{
      PTENS_CPUONLY();
      PTENS_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	r.view2().add_matmul_TA(x.fused_view_of(i),fused_view_of(i));
    }

 
  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPack reduce0() const{
      RtensorPack R(size(),Gdims(2*nc),cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i).sum01_into(R.view1_of(i).block(0,nc));
	view_of(i).diag01().sum0_into(R.view1_of(i).block(nc,nc));
      }
      return R;
    }

    RtensorPack reduce0(const int offs, const int n) const{
      RtensorPack R(size(),Gdims(n),cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i,offs,n).sum01_into(R.view1_of(i));
	view_of(i,offs+n,n).diag01().sum0_into(R.view1_of(i));
      }
      return R;
    }

    RtensorPack reduce0(const AindexPack& list) const{
      int N=list.size();
      RtensorPack R(N,Gdims(2*nc),cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	view_of(list.tens(i),list.ix(i)).sum01_into(R.view1_of(i).block(0,nc));
	view_of(list.tens(i),list.ix(i)).diag01().sum0_into(R.view1_of(i).block(nc,nc));
      }
      return R;
    }

    RtensorPack reduce0(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      RtensorPack R(N,Gdims(n),cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	view_of(list.tens(i),list.ix(i),offs,n).sum01_into(R.view1_of(i));
	view_of(list.tens(i),list.ix(i),offs+n,n).diag01().sum0_into(R.view1_of(i));
      }
      return R;
    }

    RtensorPack reduce1() const{
      cnine::array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),3*nc}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i).sum0_into(R.view2_of(i).block(0,0,-1,nc));
	view_of(i).sum1_into(R.view2_of(i).block(0,nc,-1,nc));
	R.view2_of(i).block(0,2*nc,-1,nc)+=view_of(i).diag01();
      }
      return R;
    }

    RtensorPack reduce1(const int offs, const int n) const{
      cnine::array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),n}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i,offs,n).sum0_into(R.view2_of(i));
	view_of(i,offs+n,n).sum1_into(R.view2_of(i));
	R.view2_of(i)+=view_of(i,offs+2*n,n).diag01();
      }
      return R;
    }

    RtensorPack reduce1(const AindexPack& list) const{
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),3*nc}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	view_of(list.tens(i),list.ix(i)).sum0_into(R.view2_of(i).block(0,0,-1,nc));
	view_of(list.tens(i),list.ix(i)).sum1_into(R.view2_of(i).block(0,nc,-1,nc));
	R.view2_of(i).block(0,2*nc,-1,nc)+=view_of(list.tens(i),list.ix(i)).diag01();
      }
      return R;
    }

    RtensorPack reduce1(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),n}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	view_of(list.tens(i),list.ix(i),offs,n).sum0_into(R.view2_of(i));
	view_of(list.tens(i),list.ix(i),offs+n,n).sum1_into(R.view2_of(i));
	R.view2_of(i)+=view_of(list.tens(i),list.ix(i),offs+2*n,n).diag01();
      }
      return R;
    }

    RtensorPack reduce2() const{
      return *this;
    }

    RtensorPack reduce2(const int offs, const int n) const{ // flipping 
      cnine::array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),k_of(i),n}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++){
	R.view3_of(i)+=view_of(i,offs,n);
	R.view3_of(i)+=view_of(i,offs+n,n).transp01();
      }
      return R;
    }

    RtensorPack reduce2(const AindexPack& list) const{ // no flipping 
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),list.nix(i),nc}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	R.view3_of(i)+=view_of(list.tens(i),list.ix(i));
      }
      return R;
    }

    RtensorPack reduce2(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      cnine::array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),list.nix(i),n}));
      RtensorPack R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs,n);
	R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs+n,n).transp();
      }
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const RtensorPack& x){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++){
	view_of(i)+=repeat0(repeat0(x.view1_of(i).block(0,nc),k_of(i)),k_of(i));
	view_of(i).diag01()+=repeat0(x.view1_of(i).block(nc,nc),k_of(i));
      }
    }

    void broadcast0(const RtensorPack& x, const int offs){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(repeat0(x.view1_of(i),k_of(i)),k_of(i));
	view_of(i,offs+n,n).diag01()+=repeat0(x.view1_of(i),k_of(i));
      }
    }

    void broadcast0(const RtensorPack& x, const AindexPack& list){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++){
	if(x.dim_of(i,0)==0) continue;
	view_of(list.tens(i),list.ix(i))+=repeat0(repeat0(x.view1_of(i).block(0,nc),list.nix(i)),list.nix(i));
	view_of(list.tens(i),list.ix(i)).diag01()+=repeat0(x.view1_of(i).block(nc,nc),list.nix(i));
      }
    }

    void broadcast0(const RtensorPack& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++){
	if(x.dim_of(i,0)==0) continue; // probably redundant
	view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(repeat0(x.view1_of(i),list.nix(i)),list.nix(i));
	view_of(list.tens(i),list.ix(i),offs+n,n).diag01()+=repeat0(x.view1_of(i),list.nix(i));
      }
    }

    void broadcast1(const RtensorPack& x){
      for(int i=0; i<size(); i++){
	view_of(i)+=repeat0(x.view2_of(i).block(0,0,-1,nc),k_of(i));
	view_of(i)+=repeat1(x.view2_of(i).block(0,nc,-1,nc),k_of(i));
	view_of(i).diag01()+=x.view2_of(i).block(0,2*nc,-1,nc);
      }
    }

    void broadcast1(const RtensorPack& x, const int offs){
      const int n=x.dim_of(0,1);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(x.view2_of(i),k_of(i));
	view_of(i,offs+n,n)+=repeat1(x.view2_of(i),k_of(i));
	view_of(i,offs+2*n,n).diag01()+=x.view2_of(i);
      }
    }

    void broadcast1(const RtensorPack& x, const AindexPack& list){
      int N=list.size();
      //const int n=x.dim_of(0,1);
      for(int i=0; i<N; i++){
	if(x.dim_of(i,0)==0) continue;
	view_of(list.tens(i),list.ix(i))+=repeat0(x.view2_of(i).block(0,0,-1,nc),list.nix(i));
	view_of(list.tens(i),list.ix(i))+=repeat1(x.view2_of(i).block(0,nc,-1,nc),list.nix(i));
	view_of(list.tens(i),list.ix(i)).diag01()+=x.view2_of(i).block(0,2*nc,-1,nc);
      }
    }

    void broadcast1(const RtensorPack& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,1);
      for(int i=0; i<N; i++){
	if(x.dim_of(i,0)==0) continue;
	view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view2_of(i),list.nix(i));
	view_of(list.tens(i),list.ix(i),offs+n,n)+=repeat1(x.view2_of(i),list.nix(i));
	view_of(list.tens(i),list.ix(i),offs+2*n,n).diag01()+=x.view2_of(i);
      }
    }

    void broadcast2(const RtensorPack& x){ // no flipping
      //const int n=x.dim_of(0,2);
      for(int i=0; i<size(); i++){
	view_of(i)+=x.view3_of(i);
      }
    }

    void broadcast2(const RtensorPack& x, const int offs){
      const int n=x.dim_of(0,2);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=x.view3_of(i);
	view_of(i,offs+n,n)+=x.view3_of(i).transp01();
      }
    }

    void broadcast2(const RtensorPack& x, const AindexPack& list){
      int N=list.size();
      //const int n=x.dim_of(0,2);
      for(int i=0; i<N; i++){
	if(x.dim_of(i,0)==0) continue;
	view_of(list.tens(i),list.ix(i))+=x.view3_of(i);
      }
    }

    void broadcast2(const RtensorPack& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,2);
      for(int i=0; i<N; i++){
	if(x.dim_of(i,0)==0) continue;
	view_of(list.tens(i),list.ix(i),offs,n)+=x.view3_of(i);
	view_of(list.tens(i),list.ix(i),offs+n,n)+=x.view3_of(i).transp01();
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors2";
    }

    string str(const string indent="") const{
      if(dev>0){
	Ptensors2 y(*this,0);
	return y.str();
      }
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors2& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
