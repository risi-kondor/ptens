#ifndef _ptens_Ptensor2pack
#define _ptens_Ptensor2pack

#include "Rtensor3_view.hpp"
#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor2.hpp"


namespace ptens{


  class Ptensor2pack: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc;
    AtomsPack atoms;

    ~Ptensor2pack(){
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensor2pack(){}

    Ptensor2pack(const int _nc, const int _dev=0):
      RtensorPool(_dev), nc(_nc){}


  public: // ----- Named constructors ------------------------------------------------------------------------


    static Ptensor2pack raw(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor2pack R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor2::raw(_atoms(i),_nc));
      }
      return R;
    }


    static Ptensor2pack zero(const AtomsPack& _atoms, const int _nc, const int _dev=0){
      Ptensor2pack R(_nc,_dev);
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor2::zero(_atoms(i),_nc));
      }
      return R;
    }


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensor2pack(const Ptensor2pack& x):
      RtensorPool(x),
      atoms(x.atoms){}
	
    Ptensor2pack(Ptensor2pack&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)){}

    Ptensor2pack& operator=(const Ptensor2pack& x)=delete;


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }

    int push_back(const Ptensor2& x){
      if(size()==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }


  public: // ---- Message passing ----------------------------------------------------------------------------

    /*
    Ptensor2pack fwd(const Cgraph& graph) const{
      Ptensor2pack R;
      for(int i=0; i<graph.maxj; i++) //TODO
	R.push_back(Ptensor2::zero(atoms_of(i),5*2));
      R.forwardMP(*this,graph);
      return R;
    }


    void forwardMP(const Ptensor2pack& x, const Cgraph& graph){
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


  public: // ---- Reductions ---------------------------------------------------------------------------------


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


  public: // ---- Broadcasting -------------------------------------------------------------------------------

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

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Ptensor "<<i<<" "<<Atoms(atoms(i))<<":"<<endl;
	oss<<RtensorPool::operator()(i).str()<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor2pack& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
