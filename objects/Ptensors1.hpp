#ifndef _ptens_Ptensors1
#define _ptens_Ptensors1

#include "Cgraph.hpp"
//#include "Ptensor1subpack.hpp"
//#include "PtensorSubpackSpecializer.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor1.hpp"

#include "Ptensors0.hpp"


namespace ptens{


  class Ptensors1: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc;
    AtomsPack atoms;
    bool is_view=false;


#ifdef WITH_FAKE_GRAD
    Ptensors1* grad=nullptr;
#endif 

    ~Ptensors1(){
#ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
#endif 
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensors1(){}

    Ptensors1(const int _nc, const int _dev=0):
      RtensorPool(_dev), nc(_nc){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors1(const int _n, const int _k, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_n,{_k,_nc},dummy,_dev), atoms(_n,_k), nc(_nc){}


  public: // ----- Constructors ------------------------------------------------------------------------------


    static Ptensors1 raw(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors1(_n,_k,_nc,cnine::fill_raw(),_dev);}

    static Ptensors1 zero(const int _n, const int _k, const int _nc, const int _dev=0){
      return Ptensors1(_n,_k,_nc,cnine::fill_zero(),_dev);}

    static Ptensors1 sequential(const int _n, const int _k, const int _nc, const int _dev=0){
      Ptensors1 R(_n,_k,_nc,cnine::fill_raw(),_dev);
      for(int i=0; i<_n; i++)
	R.view2_of(i).set(i); 
      return R;
    }


    static Ptensors1 raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor1::raw(_atoms(i),_nc));
      }
      return R;
    }

    static Ptensors1 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor1::zero(_atoms(i),_nc));
      }
      return R;
    }

    static Ptensors1 gaussian(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor1::gaussian(_atoms(i),_nc));
      }
      return R;
    }

    static Ptensors1 sequential(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors1 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor1::sequential(_atoms(i),_nc));
      }
      return R;
    }


  public: // ---- Spawning -----------------------------------------------------------------------------------

    //static Ptensors0 zeros_like(const Ptenso

    static Ptensors1* new_zeros_like(const Ptensors1& x){
      return new Ptensors1(RtensorPool::zeros_like(x),x.atoms,x.nc);
    }

    
  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors1(RtensorPool&& x, const AtomsPack& _atoms, const int _nc):
      RtensorPool(std::move(x)), atoms(_atoms), nc(_nc){}


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors1(const Ptensors1& x):
      RtensorPool(x),
      atoms(x.atoms),
      nc(x.nc){}
	
    Ptensors1(Ptensors1&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)),
      nc(x.nc){}

    Ptensors1& operator=(const Ptensors1& x)=delete;


  public: // ---- Experimental -------------------------------------------------------------------------------


    #ifdef WITH_FAKE_GRAD
    //void add_to_grad(const Ptensors1& x){
    //if(grad) grad->add(x);
    //else grad=new Ptensors1(x);
    //}

    Ptensors1& get_grad(){
      if(!grad) grad=Ptensors1::new_zeros_like(*this);
      return *grad;
    }

    Ptensors1* get_gradp(){
      if(!grad) grad=Ptensors1::new_zeros_like(*this);
      return grad;
    }

    Ptensors1* gradp(){
      if(!grad) grad=Ptensors1::new_zeros_like(*this);
      return grad;
    }

    //Ptensors1 view_of_grad(){
    //if(!grad) grad=new_zeros_like(*this);
    //return grad->view();
    //}
    #endif 


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }

    AtomsPack& get_atomsref(){
      return atoms;
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

    Rtensor2_view view_of(const int i) const{
      return RtensorPool::view2_of(i);
    }

    Rtensor2_view view_of(const int i, const int offs, const int n) const{
      return RtensorPool::view2_of(i).block(0,offs,-1,n);
    }

    Ptensor1_xview view_of(const int i, const vector<int>& ix) const{
      vector<int> v=headers(i);
      assert(v.size()==3);
      if(dev==1) return Ptensor1_xview(arrg+v[0],v[2],v[2],1,ix,1);
      return Ptensor1_xview(arr+v[0],v[2],v[2],1,ix,0);
    }

    Ptensor1_xview view_of(const int i, const vector<int>& ix, const int offs, const int n) const{
      vector<int> v=headers(i);
      assert(v.size()==3);
      if(dev==1) return Ptensor1_xview(arrg+v[0]+offs,n,v[2],1,ix,1);
      return Ptensor1_xview(arr+v[0]+offs,n,v[2],1,ix,0);
    }

    Ptensor1 operator()(const int i) const{
      return Ptensor1(tensor_of(i),atoms_of(i));
    }

    int push_back(const Ptensor1& x){
      if(size()==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_mprod(const Ptensors1& x, const rtensor& y){
      assert(x.size()==size());
      for(int i=0; i<size(); i++)
	view_of_tensor(i).add_mprod(x.view_of_tensor(i),y);
    }

 
  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool reduce0() const{
      RtensorPool R(size(),Gdims(nc),cnine::fill_zero());
      for(int i=0; i<size(); i++)
	view_of(i).sum0_into(R.view1_of(i));
      return R;
    }

    RtensorPool reduce0(const int offs, const int n) const{
      RtensorPool R(size(),Gdims(nc),cnine::fill_zero());
      for(int i=0; i<size(); i++)
	view_of(i,offs,n).sum0_into(R.view1_of(i));
      return R;
    }

    RtensorPool reduce0(const AindexPack& list) const{
      int N=list.size();
      RtensorPool R(N,Gdims(nc),cnine::fill_zero());
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i)).sum0_into(R.view1_of(i));
      return R;
    }

    RtensorPool reduce0(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      RtensorPool R(N,Gdims(nc),cnine::fill_zero());
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i),offs,n).sum0_into(R.view1_of(i));
      return R;
    }


    RtensorPool reduce1() const{
      return *this;
    }

    RtensorPool reduce1(const int offs, const int n) const{
      array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),n}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++)
	R.view2_of(i)+=view_of(i,offs,n);
      return *this;
    }

    RtensorPool reduce1(const AindexPack& list) const{
      int N=list.size();
      array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),nc});
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++)
	R.view2_of(i)+=view_of(list.tens(i),list.ix(i));
      return R;
    }

    RtensorPool reduce1(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back({list.nix(i),n});
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++)
	R.view2_of(i)+=view_of(list.tens(i),list.ix(i),offs,n);
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const RtensorPool& x){
      for(int i=0; i<size(); i++){
	view_of(i)+=repeat0(x.view1_of(i),k_of(i));
      }
    }

    void broadcast0(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(x.view1_of(i),k_of(i));
      }
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list){
      int N=list.size();
      for(int i=0; i<N; i++)
	view_of(list.tens(i))+=repeat0(x.view1_of(i),list.nix(i));
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view1_of(i),list.nix(i));
    }


    void broadcast1(const RtensorPool& x){
      for(int i=0; i<size(); i++){
	view_of(i)+=x.view2_of(i);
      }
    }

    void broadcast1(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,1);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=x.view2_of(i);
      }
    }

    void broadcast1(const RtensorPool& x, const AindexPack& list){
      int N=list.size();
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i))+=x.view2_of(i);
    }

    void broadcast1(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,1);
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i),offs,n)+=x.view2_of(i);
    }




  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<(*this)(i)<<endl;
	//oss<<indent<<"Ptensor "<<i<<" "<<Atoms(atoms(i))<<":"<<endl;
	//oss<<RtensorPool::operator()(i).str()<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensors1& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
    /*
    RtensorPool messages0(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++){
	Rtensor2_view source=view2_of(src_list.tix(i));
	Rtensor1_view dest=R.view1_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=dest.n0;

	for(int c=0; c<nc; c++){
	  float t=0; 
	  for(int j=0; j<n; j++) 
	    t+=source(ix[j],c);
	  dest.set(c,t);
	}
      }

      return R;
    }


    RtensorPool messages1(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({src_list.nindices(i),dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++){
	Rtensor2_view source=view2_of(src_list.tix(i));
	Rtensor2_view dest=R.view2_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=dest.n1;

	for(int c=0; c<nc; c++){
	  for(int j=0; j<n; j++) 
	    dest.set(j,c,source(ix[j],c));
	}
      }

      return R;
    }
    */
    /*
    Ptensors1 fwd(const Cgraph& graph) const{
      Ptensors1 R;
      for(int i=0; i<graph.m; i++) //TODO
	R.push_back(Ptensor1::zero(atoms_of(i),5*2));
      R.forwardMP(*this,graph);
      return R;
    }


    void forwardMP(const Ptensors1& x, const Cgraph& graph){
      AindexPack src_indices;
      AindexPack dest_indices;

      graph.forall_edges([&](const int i, const int j){
	  Atoms atoms0=atoms_of(i);
	  Atoms atoms1=atoms_of(j);
	  Atoms intersect=atoms0.intersect(atoms1);
	  src_indices.push_back(i,atoms0(intersect));
	  dest_indices.push_back(j,atoms1(intersect));
	});

      RtensorPool messages0=x.messages0(src_indices);
      add_messages0(messages0,dest_indices,0);
      //cout<<messages0<<endl;

      RtensorPool messages1=x.messages1(src_indices);
      add_messages1(messages1,dest_indices,5); // TODO 
      //cout<<messages1<<endl;

    }
    */
    /*
    void add_linmaps(const Ptensors0& x, const int offs=0){
      assert(x.size()==size());
      assert(offs+x.nc<=nc);
      int _nc=x.nc;
      for(int i=0; i<size(); i++){
	for(int c=0; c<_nc; c++)
	  view2_of(i).slice1(offs+c).add(x.view1_of(i)(c));
	}
    }


    void add_linmaps_to(Ptensors0& x, const int offs=0) const{
      assert(x.size()==size());
      assert(offs+nc<=x.nc);
      for(int i=0; i<size(); i++){
	for(int c=0; c<nc; c++)
	  x.view1_of(i).inc(c+offs,view2_of(i).slice1(c).sum());
	}
    }


    void add_linmaps(const Ptensors1& x, const int offs=0){
      assert(x.size()==size());
      assert(offs+2*x.nc<=nc);
      int _nc=x.nc;
      for(int i=0; i<size(); i++){
	int k=x.k_of(i);
	assert(k==k_of(i));
	view2_of(i).block(0,offs,k,_nc)+=x.view2_of(i);
	for(int c=0; c<_nc; c++){
	  float t=0; 
	  for(int j=0; j<k; j++)
	    t+=x.view2_of(i)(j,c);
	  for(int j=0; j<k; j++)
	    view2_of(i).inc(j,offs+c+_nc,t);
	}
      }
    }
    */

    /*
    void add_messages0(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++){
	Rtensor1_view source=messages.view1_of(i);
	Rtensor2_view dest=view2_of(dest_list.tix(i));
	vector<int> ix=dest_list.indices(i);
	int n=ix.size();
	int nc=source.n0;

	for(int c=0; c<nc; c++){
	  float v=source(c);
	  for(int j=0; j<n; j++) 
	    dest.inc(ix[j],c+coffs,v);
	}
      }
    }


    void add_messages1(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++){
	Rtensor2_view source=messages.view2_of(i);
	Rtensor2_view dest=view2_of(dest_list.tix(i));
	vector<int> ix=dest_list.indices(i);
	int n=ix.size();
	int nc=source.n1;

	for(int c=0; c<nc; c++){
	  for(int j=0; j<n; j++) 
	    dest.inc(ix[j],c+coffs,source(j,c));
	}
      }
    }
    */
