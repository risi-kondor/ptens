#ifndef _ptens_Ptensors0
#define _ptens_Ptensors0

#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor0.hpp"
#include "loose_ptr.hpp"
#include "diff_class.hpp"


namespace ptens{


  class Ptensors0: public RtensorPool, public diff_class<Ptensors0>{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc=0;
    AtomsPack atoms;
    bool is_view=false;


    ~Ptensors0(){
#ifdef WITH_FAKE_GRAD
      if(!is_view && grad) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors0(const int _nc, const int _dev=0):
      RtensorPool(_dev), nc(_nc){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_n, cnine::Gdims({_nc}), dummy, _dev), atoms(_n), nc(_nc){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_atoms.size(), cnine::Gdims({_nc}), dummy, _dev), atoms(_atoms), nc(_nc){
    }


  public: // ----- Named Constructors ------------------------------------------------------------------------


    static Ptensors0 raw(const int _n, const int _nc, const int _dev=0){
      return Ptensors0(_n,_nc,cnine::fill_raw(),_dev);}

    static Ptensors0 zero(const int _n, const int _nc, const int _dev=0){
      return Ptensors0(_n,_nc,cnine::fill_zero(),_dev);}

    static Ptensors0 sequential(const int _n, const int _nc, const int _dev=0){
      Ptensors0 R(_n,_nc,cnine::fill_raw(),_dev);
      for(int i=0; i<_n; i++) R.view1_of(i).set(i);
      return R;
    }

    static Ptensors0 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_raw(),_dev);}

    static Ptensors0 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_zero(),_dev);}

    static Ptensors0 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      return Ptensors0(_atoms,_nc,cnine::fill_gaussian(),_dev);}

    static Ptensors0 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors0 R(_atoms,_nc,cnine::fill_raw(),_dev);
      for(int i=0; i<R.size(); i++) R.view1_of(i).set(i);
      return R;
    }


    static Ptensors0 concat(const Ptensors0& x, const Ptensors0& y){
      Ptensors0 R=Ptensors0::zero(x.atoms,x.nc+y.nc);
      R.add_to_channels(x,0);
      R.add_to_channels(y,x.nc);
      return R;
    }


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static Ptensors0* new_zeros_like(const Ptensors0& x){
      return new Ptensors0(RtensorPool::zeros_like(x),x.atoms,x.nc);
    }

    
  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors0(RtensorPool&& x, const AtomsPack& _atoms, const int _nc):
      RtensorPool(std::move(x)), atoms(_atoms), nc(_nc){}

    Ptensors0(const rtensor& A):
      RtensorPool(A), atoms(A.dim(0)){
      nc=A.dim(1);
    }

    rtensor tensor() const{
      CNINE_CPUONLY();
      return rtensor({size(),nc},arr,0);
    }

    #ifdef _WITH_ATEN
    Ptensors0(const at::Tensor& T):
      RtensorPool(rtensor(T)){
      assert(size()>0);
      atoms=AtomsPack(size());
      nc=dim_of(0,0);
    }
    #endif 


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x):
      RtensorPool(x),
      diff_class<Ptensors0>(x),
      atoms(x.atoms),
      nc(x.nc){
      PTENS_COPY_WARNING();

      #ifdef WITH_FAKE_GRAD
      //if(x.grad) grad=new Ptensors0(*grad);
      #endif 
    }
	
    Ptensors0(Ptensors0&& x):
      RtensorPool(std::move(x)),
      diff_class<Ptensors0>(std::move(x)),
      atoms(std::move(x.atoms)),
      nc(x.nc){
      PTENS_MOVE_WARNING();

      #ifdef WITH_FAKE_GRAD
      //grad=x.grad;
      //x.grad=nullptr;
      #endif 
    }

    Ptensors0& operator=(const Ptensors0& x)=delete;


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
      return RtensorPool::operator()(i);
    }

    Rtensor1_view view_of(const int i) const{
      return RtensorPool::view1_of(i);
    }

    Rtensor1_view view_of(const int i, const int offs, const int n) const{
      return RtensorPool::view1_of(i).block(offs,n);
    }

    Rtensor1_view view_of(const int i, const vector<int>& ix) const{
      return RtensorPool::view1_of(i);
    }

    Rtensor1_view view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      return RtensorPool::view1_of(i).block(offs,n);
    }

    Ptensor0 operator()(const int i) const{
      return Ptensor0(tensor_of(i),atoms_of(i));
    }

    void push_back(const Ptensor0& x){
      if(nc==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_to_channels(const Ptensors0& x, const int offs){
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i,offs,x.nc)+=x.view_of(i);
    }

    void add_channels(const Ptensors0& x, const int offs){
      int N=size();
      PTENS_ASSRT(x.size()==N);
      for(int i=0; i<N; i++)
	view_of(i)+=x.view_of(i,offs,x.nc);
    }

    void add_mprod(const Ptensors0& x, const rtensor& y){
      PTENS_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	add_matmul_Ax_to(view_of(i),y.view2().transp(),x.view_of(i));
    }

    void add_mprod_back0(const Ptensors0& g, const rtensor& y){
      PTENS_ASSRT(g.size()==size());
      for(int i=0; i<size(); i++)
	add_matmul_Ax_to(view_of(i),y.view2(),g.view_of(i));
    }

    void add_mprod_back1_to(rtensor& r, const Ptensors0& x) const{
      PTENS_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	r.view2().add_outer(x.view_of(i),view_of(i));
    }
 
 
  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool reduce0() const{
      RtensorPool R(size(),Gdims(nc),cnine::fill_zero());
      for(int i=0; i<size(); i++)
	R.view1_of(i).add(view_of(i));
      return R;
    }

    RtensorPool reduce0(const int offs, const int n) const{
      RtensorPool R(size(),Gdims(n),cnine::fill_zero());
      for(int i=0; i<size(); i++)
	R.view1_of(i).add(view_of(i,offs,n));
      return R;
    }

    RtensorPool reduce0(const AindexPack& list) const{
      int N=list.size();
      array_pool<int> dims;
      RtensorPool R(N,Gdims(nc),cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	R.view1_of(i)=view_of(list.tix(i)); // OK
      }
      return R;
    }

    RtensorPool reduce0(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      RtensorPool R(N,Gdims(nc),cnine::fill_zero());
      for(int i=0; i<N; i++){
	if(list.nix(i)==0) continue;
	R.view1_of(i)=view_of(list.tix(i),offs,n); // OK
      }
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------

    
    void broadcast0(const RtensorPool& x){
      for(int i=0; i<size(); i++)
	view_of(i)+=x.view1_of(i);
    }

    void broadcast0(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++)
	view_of(i,offs,n).add(x.view1_of(i));
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list){
      int N=list.size();
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i))+=x.view1_of(i);
	cout<<"zzzzzz"<<view_of(list.tens(i))<<endl;
      }
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i),offs,n)+=x.view1_of(i);
    }



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors0";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors0& x){
      stream<<x.str(); return stream;}

  };

}


#endif 

    //Ptensors0* gradp(){
    //if(!grad) grad=Ptensors0::new_zeros_like(*this);
    //return grad;
    //}

    //Ptensors0* get_gradp(){
    //if(!grad) grad=Ptensors0::new_zeros_like(*this);
    //return grad;
    //}

    //Ptensors1 view_of_grad(){
    //if(!grad) grad=new_zeros_like(*this);
    //return grad->view();
    //}
    //void add_to_grad(const Ptensors0* x){
    //if(grad) grad->add(*x);
    //else grad=new Ptensors0(*x);
    //}

    /*
    void add_to_grad(const Ptensors0& x){
      if(grad) grad->add(x);
      else grad=new Ptensors0(x);
    }

    Ptensors0& get_grad(){
      if(!grad) grad=Ptensors0::new_zeros_like(*this);
      return *grad;
    }

    loose_ptr<Ptensors0> get_gradp(){
      if(!grad) grad=Ptensors0::new_zeros_like(*this);
      return grad;
    }
    */
