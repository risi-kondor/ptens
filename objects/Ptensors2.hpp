#ifndef _ptens_Ptensors2
#define _ptens_Ptensors2

#include "Rtensor3_view.hpp"
#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor2.hpp"
#include "diff_class.hpp"


namespace ptens{


  class Ptensors2: public RtensorPool, public diff_class<Ptensors2>{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
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
      RtensorPool(_dev), nc(_nc){}

   template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors2(const int _n, const int _k, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_n,{_k,_k,_nc},dummy,_dev), atoms(_n,_k), nc(_nc){}


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


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static Ptensors2* new_zeros_like(const Ptensors2& x){
      return new Ptensors2(RtensorPool::zeros_like(x),x.atoms,x.nc);
    }

    
  public: // ----- Conversions -------------------------------------------------------------------------------


    Ptensors2(RtensorPool&& x, const AtomsPack& _atoms, const int _nc):
      RtensorPool(std::move(x)), atoms(_atoms), nc(_nc){}


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors2(const Ptensors2& x):
      RtensorPool(x),
      diff_class<Ptensors2>(x),
      atoms(x.atoms),
      nc(x.nc){
      PTENS_COPY_WARNING();
    }
	
    Ptensors2(Ptensors2&& x):
      RtensorPool(std::move(x)),
      diff_class<Ptensors2>(std::move(x)),
      atoms(std::move(x.atoms)),
      nc(x.nc){
      PTENS_COPY_WARNING();
    }

    Ptensors2& operator=(const Ptensors2& x)=delete;


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

    Rtensor3_view view_of(const int i) const{
      return RtensorPool::view3_of(i);
    }

    Rtensor2_view fused_view_of(const int i) const{
      return RtensorPool::view3_of(i).fuse01();
    }

    Rtensor3_view view_of(const int i, const int offs, const int n) const{
      return RtensorPool::view3_of(i).block(0,0,offs,-1,-1,n);
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
      RtensorPool::push_back(x);
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
      PTENS_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	fused_view_of(i).add_matmul_AA(x.fused_view_of(i),y.view2());
    }

    void add_mprod_back0(const Ptensors2& g, const rtensor& y){
      PTENS_ASSRT(g.size()==size());
      for(int i=0; i<size(); i++)
	fused_view_of(i).add_matmul_AT(g.fused_view_of(i),y.view2());
    }

    void add_mprod_back1_to(rtensor& r, const Ptensors2& x) const{
      PTENS_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	r.view2().add_matmul_TA(x.fused_view_of(i),fused_view_of(i));
    }

 
  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool reduce0() const{
      RtensorPool R(size(),Gdims(2*nc),cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i).sum01_into(R.view1_of(i).block(0,nc));
	view_of(i).diag01().sum0_into(R.view1_of(i).block(nc,nc));
      }
      return R;
    }

    RtensorPool reduce0(const int offs, const int n) const{
      RtensorPool R(size(),Gdims(n),cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i,offs,n).sum01_into(R.view1_of(i));
	view_of(i,offs+n,n).diag01().sum0_into(R.view1_of(i));
      }
      return R;
    }

    RtensorPool reduce0(const AindexPack& list) const{
      int N=list.size();
      RtensorPool R(N,Gdims(2*nc),cnine::fill_zero());
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i)).sum01_into(R.view1_of(i).block(0,nc));
	view_of(list.tens(i),list.ix(i)).diag01().sum0_into(R.view1_of(i).block(nc,nc));
      }
      return R;
    }

    RtensorPool reduce0(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      RtensorPool R(N,Gdims(n),cnine::fill_zero());
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i),offs,n).sum01_into(R.view1_of(i));
	view_of(list.tens(i),list.ix(i),offs+n,n).diag01().sum0_into(R.view1_of(i));
      }
      return R;
    }

    RtensorPool reduce1() const{
      array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),3*nc}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i).sum0_into(R.view2_of(i).block(0,0,-1,nc));
	view_of(i).sum1_into(R.view2_of(i).block(0,nc,-1,nc));
	R.view2_of(i).block(0,2*nc,-1,nc)+=view_of(i).diag01();
      }
      return R;
    }

    RtensorPool reduce1(const int offs, const int n) const{
      array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),n}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++){
	view_of(i,offs,n).sum0_into(R.view2_of(i));
	view_of(i,offs+n,n).sum1_into(R.view2_of(i));
	R.view2_of(i)+=view_of(i,offs+2*n,n).diag01();
      }
      return R;
    }

    RtensorPool reduce1(const AindexPack& list) const{
      int N=list.size();
      array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),3*nc}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i)).sum0_into(R.view2_of(i).block(0,0,-1,nc));
	view_of(list.tens(i),list.ix(i)).sum1_into(R.view2_of(i).block(0,nc,-1,nc));
	R.view2_of(i).block(0,2*nc,-1,nc)+=view_of(list.tens(i),list.ix(i)).diag01();
      }
      return R;
    }

    RtensorPool reduce1(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),n}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i),offs,n).sum0_into(R.view2_of(i));
	view_of(list.tens(i),list.ix(i),offs+n,n).sum1_into(R.view2_of(i));
	R.view2_of(i)+=view_of(list.tens(i),list.ix(i),offs+2*n,n).diag01();
      }
      return R;
    }

    RtensorPool reduce2() const{
      return *this;
    }

    RtensorPool reduce2(const int offs, const int n) const{ // flipping 
      array_pool<int> dims;
      for(int i=0; i<size(); i++)
	dims.push_back(vector<int>({k_of(i),k_of(i),n}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<size(); i++){
	R.view3_of(i)+=view_of(i,offs,n);
	R.view3_of(i)+=view_of(i,offs+n,n).transp01();
      }
      return R;
    }

    RtensorPool reduce2(const AindexPack& list) const{ // no flipping 
      int N=list.size();
      array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),list.nix(i),nc}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	R.view3_of(i)+=view_of(list.tens(i),list.ix(i));
      }
      return R;
    }

    RtensorPool reduce2(const AindexPack& list, const int offs, const int n) const{
      int N=list.size();
      array_pool<int> dims;
      for(int i=0; i<N; i++)
	dims.push_back(vector<int>({list.nix(i),list.nix(i),nc}));
      RtensorPool R(dims,cnine::fill_zero());
      for(int i=0; i<N; i++){
	R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs,n);
	R.view3_of(i)+=view_of(list.tens(i),list.ix(i),offs+n,n).transp01();
      }
      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const RtensorPool& x){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++){
	view_of(i)+=repeat0(repeat0(x.view1_of(i).block(0,nc),k_of(i)),k_of(i));
	view_of(i).diag01()+=repeat0(x.view1_of(i).block(nc,nc),k_of(i));
      }
    }

    void broadcast0(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(repeat0(x.view1_of(i),k_of(i)),k_of(i));
	view_of(i,offs+n,n).diag01()+=repeat0(x.view1_of(i),k_of(i));
      }
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i))+=repeat0(repeat0(x.view1_of(i).block(0,nc),list.nix(i)),list.nix(i));
	view_of(list.tens(i),list.ix(i)).diag01()+=repeat0(x.view1_of(i).block(nc,nc),list.nix(i));
      }
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(repeat0(x.view1_of(i),list.nix(i)),list.nix(i));
	view_of(list.tens(i),list.ix(i),offs+n,n).diag01()+=repeat0(x.view1_of(i),list.nix(i));
      }
    }

    void broadcast1(const RtensorPool& x){
      for(int i=0; i<size(); i++){
	view_of(i)+=repeat0(x.view2_of(i).block(0,0,-1,nc),k_of(i));
	view_of(i)+=repeat1(x.view2_of(i).block(0,nc,-1,nc),k_of(i));
	view_of(i).diag01()+=x.view2_of(i).block(0,2*nc,-1,nc);
      }
    }

    void broadcast1(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,1);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(x.view2_of(i),k_of(i));
	view_of(i,offs+n,n)+=repeat1(x.view2_of(i),k_of(i));
	view_of(i,offs+2*n,n).diag01()+=x.view2_of(i);
      }
    }

    void broadcast1(const RtensorPool& x, const AindexPack& list){
      int N=list.size();
      //const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i))+=repeat0(x.view2_of(i).block(0,0,-1,nc),list.nix(i));
	view_of(list.tens(i),list.ix(i))+=repeat1(x.view2_of(i).block(0,nc,-1,nc),list.nix(i));
	view_of(list.tens(i),list.ix(i)).diag01()+=x.view2_of(i).block(0,2*nc,-1,nc);
      }
    }

    void broadcast1(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,1);
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i),offs,n)+=repeat0(x.view2_of(i),list.nix(i));
	view_of(list.tens(i),list.ix(i),offs+n,n)+=repeat1(x.view2_of(i),list.nix(i));
	view_of(list.tens(i),list.ix(i),offs+2*n,n).diag01()+=x.view2_of(i);
      }
    }

    void broadcast2(const RtensorPool& x){ // no flipping
      //const int n=x.dim_of(0,2);
      for(int i=0; i<size(); i++){
	view_of(i)+=x.view3_of(i);
      }
    }

    void broadcast2(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,2);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=x.view3_of(i);
	view_of(i,offs+n,n)+=x.view3_of(i).transp01();
      }
    }

    void broadcast2(const RtensorPool& x, const AindexPack& list){
      int N=list.size();
      const int n=x.dim_of(0,2);
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i))+=x.view3_of(i);
      }
    }

    void broadcast2(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,2);
      for(int i=0; i<N; i++){
	view_of(list.tens(i),list.ix(i),offs,n)+=x.view3_of(i);
	view_of(list.tens(i),list.ix(i),offs+n,n)+=x.view3_of(i).transp01();
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors2";
    }

    string str(const string indent="") const{
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
    /*
    RtensorPool messages0(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++){
	Rtensor3_view source=view3_of(src_list.tix(i));
	Rtensor1_view dest=R.view1_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=source.n2;
	assert(dest.n0==2*nc);

	for(int c=0; c<nc; c++){
	  float t=0; 
	  for(int j0=0; j0<n; j0++) 
	    for(int j1=0; j1<n; j1++) 
	      t+=source(ix[j0],ix[j1],c);
	  dest.set(c,t);
	}

	for(int c=0; c<nc; c++){
	  float t=0; 
	  for(int j0=0; j0<n; j0++) 
	    t+=source(ix[j0],ix[j0],c);
	  dest.set(c+nc,t);
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
	Rtensor3_view source=view3_of(src_list.tix(i));
	Rtensor2_view dest=R.view2_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=source.n2;
	assert(dest.n1==3*nc);

	for(int c=0; c<nc; c++){
	  for(int j0=0; j0<n; j0++){
	    float t=0; 
	    for(int j1=0; j1<n; j1++) 
	      t+=source(ix[j0],ix[j1],c);
	    dest.set(j0,c,t);
	  }
	}

	for(int c=0; c<nc; c++){
	  for(int j0=0; j0<n; j0++){
	    float t=0; 
	    for(int j1=0; j1<n; j1++) 
	      t+=source(ix[j1],ix[j0],c);
	    dest.set(j0,c+nc,t);
	  }
	}

	for(int c=0; c<nc; c++){
	  for(int j0=0; j0<n; j0++){
	    dest.set(j0,c+2*nc,source(ix[j0],ix[j0],c));
	  }
	}
      }

      return R;
    }


    RtensorPool messages2(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({src_list.nindices(i),dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++){
	Rtensor3_view source=view3_of(src_list.tix(i));
	Rtensor3_view dest=R.view3_of(i);
	vector<int> ix=src_list.indices(i);
	int n=ix.size();
	int nc=source.n2;
	assert(dest.n2==nc);


	for(int c=0; c<nc; c++){
	  for(int j0=0; j0<n; j0++){
	    for(int j1=0; j1<n; j1++){
	      dest.set(j0,j1,c,source(ix[j1],ix[j0],c));
	    }
	  }
	}
      }

      return R;
    }
    */
    /*
    Ptensors2 fwd(const Cgraph& graph) const{
      Ptensors2 R;
      for(int i=0; i<graph.maxj; i++) //TODO
	R.push_back(Ptensor2::zero(atoms_of(i),5*2));
      R.forwardMP(*this,graph);
      return R;
    }


    void forwardMP(const Ptensors2& x, const Cgraph& graph){
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
	  view3_of(i).slice2(offs+c).add(x.view1_of(i)(c));
	}
    }

    void add_linmaps(const Ptensors1& x, const int offs=0){
      assert(x.size()==size());
      assert(offs+3*x.nc<=nc);
      int _nc=x.nc;
      for(int i=0; i<size(); i++){
	int k=k_of(i);
	assert(x.k_of(i)==k);
	auto src=x.view2_of(i);
	auto dest=view3_of(i);
	for(int c=0; c<_nc; c++){

	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++){
	      dest.inc(a,b,offs+c,src(a,c));
	      dest.inc(a,b,offs+_nc+c,src(b,c));
	    }

	  for(int a=0; a<k; a++)
	    dest.inc(a,a,offs+2*_nc+c,src(a,c));
	}
      }
    }

    void add_linmaps(const Ptensors2& x, const int offs=0){
      assert(x.size()==size());
      int _nc=x.nc;
      assert(offs+3*_nc<=nc);

      for(int i=0; i<size(); i++){
	auto src=x.view3_of(i);
	auto dest=view3_of(i);
	int k=dest.n0;

	dest.block(0,0,offs,k,k,_nc).add(src);
	dest.block(0,0,offs+_nc,k,k,_nc).add(src.transp01());

	//auto r=(*this)(i).reductions1()

	  
	for(int c=0; c<_nc; c++){

	  for(int a=0; a<k; a++)
	    for(int b=0; b<k; b++){
	      dest.inc(a,b,offs+c,src(a,b,c));
	      dest.inc(a,b,offs+_nc+c,src(b,a,c));
	    }

	  for(int a=0; a<k; a++)
	    dest.inc(a,a,offs+2*_nc+c,src(a,a,c));
	}
      }
    }
    */

    /*
    void add_linmaps_to(Ptensors0& x, const int offs=0) const{
      assert(x.size()==size());
      assert(offs+x.nc<=nc);
      int _nc=x.nc;
      for(int i=0; i<size(); i++){
	for(int c=0; c<_nc; c++)
	  x.view1_of(i).inc(c,view2_of(i).slice1(c).sum());
	}
    }
    */
    /*
    void add_to_grad(const Ptensors2& x){
      if(grad) grad->add(x);
      else grad=new Ptensors2(x);
    }

    Ptensors2& get_grad(){
      if(!grad) grad=Ptensors2::new_zeros_like(*this);
      return *grad;
    }

    loose_ptr<Ptensors2> get_gradp(){
      if(!grad) grad=Ptensors2::new_zeros_like(*this);
      return grad;
    }
    */
