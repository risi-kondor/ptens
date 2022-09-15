#ifndef _ptens_Ptensors2
#define _ptens_Ptensors2

#include "Rtensor3_view.hpp"
#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor2.hpp"


namespace ptens{


  class Ptensors2: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc;
    AtomsPack atoms;

    ~Ptensors2(){
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
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor2::raw(_atoms(i),_nc));
      }
      return R;
    }


    static Ptensors2 zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensors2 R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor2::zero(_atoms(i),_nc));
      }
      return R;
    }


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors2(const Ptensors2& x):
      RtensorPool(x),
      atoms(x.atoms){}
	
    Ptensors2(Ptensors2&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)){}

    Ptensors2& operator=(const Ptensors2& x)=delete;


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
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

    Rtensor3_view view_of(const int i, const int offs, const int n) const{
      return RtensorPool::view3_of(i).block(0,0,offs,-1,-1,n);
    }

    Ptensor2 operator()(const int i) const{
      return Ptensor2(tensor_of(i),atoms_of(i));
    }

    int push_back(const Ptensor2& x){
      if(size()==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }


  public: // ---- Linmaps ------------------------------------------------------------------------------------



  public: // ---- Message passing ----------------------------------------------------------------------------



  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool reduce0() const{
      RtensorPool R(size(),Gdims(nc),cnine::fill_zero());
      for(int i=0; i<size(); i++)
	view_of(i).sum01_into(R.view1_of(i));
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
	R.view2_of(i).block(0,nc,-1,nc)+=view_of(i).diag01();
      }
      return R;
    }

    RtensorPool reduce2() const{
      return *this;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void broadcast0(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,0);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(repeat0(x.view1_of(i),k_of(i)),k_of(i));
	view_of(i,offs+n,n).diag01()+=repeat0(x.view1_of(i),k_of(i));
      }
    }

    void broadcast1(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,1);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=repeat0(x.view2_of(i),k_of(i));
	view_of(i,offs+n,n)+=repeat1(x.view2_of(i),k_of(i));
	view_of(i,offs+n,n).diag01()+=x.view2_of(i);
      }
    }

    void broadcast2(const RtensorPool& x, const int offs){
      const int n=x.dim_of(0,2);
      for(int i=0; i<size(); i++){
	view_of(i,offs,n)+=x.view3_of(i);
	view_of(i,offs+n,n)+=x.view3_of(i).transp01();
      }
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
