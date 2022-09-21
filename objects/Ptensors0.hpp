#ifndef _ptens_Ptensors0
#define _ptens_Ptensors0

#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor0.hpp"


namespace ptens{


  class Ptensors0: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc=0;
    AtomsPack atoms;

    ~Ptensors0(){
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    //Ptensors0(){}

    Ptensors0(const int _nc, const int _dev=0):
      RtensorPool(_dev), nc(_nc){}

    //Ptensors0(const int _nc, const int _memsize, const int _dev):
    //RtensorPool(_dev), nc(_nc){
    //reserve(_memsize);
    //}
    
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const int _n, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_n, cnine::Gdims({_nc}), dummy, _dev), atoms(_n), nc(_nc){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensors0(const AtomsPack& _atoms, const int _nc, const FILLTYPE& dummy, const int _dev=0):
      RtensorPool(_atoms.size(), cnine::Gdims({_nc}), dummy, _dev), atoms(_atoms), nc(_nc){
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


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


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensors0(const Ptensors0& x):
      RtensorPool(x),
      atoms(x.atoms),
      nc(x.nc){}
	
    Ptensors0(Ptensors0&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)),
      nc(x.nc){}

    Ptensors0& operator=(const Ptensors0& x)=delete;


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

    //Rtensor1_view view_of_tensor(const int i) const{
    //return RtensorPool::view1_of(i);
    //}

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

    RtensorPool reduce0(const AindexPack& src_list) const{
      int N=src_list.size();
      array_pool<int> dims;
      RtensorPool R(N,Gdims(nc),cnine::fill_zero());
      //for(int i=0; i<N; i++)
      //dims.push_back({dims_of(src_list.tix(i)).back()});
      //RtensorPool R(dims,cnine::fill_zero());

      for(int i=0; i<N; i++)
	R.view1_of(i)=view_of(src_list.tix(i));
      return R;
    }

    RtensorPool reduce0(const AindexPack& src_list, const int offs, const int n) const{
      int N=src_list.size();
      RtensorPool R(N,Gdims(nc),cnine::fill_zero());
      //array_pool<int> dims;
      //for(int i=0; i<N; i++)
      //dims.push_back({dims_of(src_list.tix(i)).back()});
      //RtensorPool R(dims,cnine::fill_zero());

      for(int i=0; i<N; i++)
	R.view1_of(i)=view_of(src_list.tix(i),offs,n);
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
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i))+=x.view1_of(i);
    }

    void broadcast0(const RtensorPool& x, const AindexPack& list, const int offs){
      int N=list.size();
      const int n=x.dim_of(0,0);
      for(int i=0; i<N; i++)
	view_of(list.tens(i),list.ix(i),offs,n)+=x.view1_of(i);
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

    friend ostream& operator<<(ostream& stream, const Ptensors0& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
    //Ptensors0 hom() const{
    //Ptensors0 R=Ptensors0::zero(atoms,nc,dev);
    //R.add_linmaps(*this);
    //return R;
    //}


   /*
    void add_messages0(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++)
	view1_of(dest_list.tix(i))=messages.view1_of(i);
    }
    */

//public: // ---- Linmaps -------------------------------------------------------------------------------------


    // 0 -> 0
    /*
    void add_linmaps(const Ptensors0& x, int offs=0){ 
      assert(x.size()==size());
      assert(offs+1*x.nc<=nc);
      for(int i=0; i<size(); i++){
      }
      offs+=broadcast0(reduce0(x),offs); // 1*1
    }

    void add_linmaps_back(const Ptensors0& x, int offs=0){ 
      assert(x.size()==size());
      assert(offs+1*nc<=x.nc);

    }
    */

    /*
    void add_linmaps(const Ptensors0& x, const int offs=0){
      assert(x.size()==size());
      assert(offs+x.nc<=nc);
      int _nc=x.nc;
      for(int i=0; i<size(); i++){
	view1_of(i).add(x.view1_of(i));
	//	view1_of(i).block(offs,_nc)=x.view1_of(i);
	//view1_of(i).block(offs+_nc,_nc).set(x.view1_of(i).sum());
      }
    }
    */
