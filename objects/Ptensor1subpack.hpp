#ifndef _ptens_Ptensor1subpack
#define _ptens_Ptensor1subpack

#include "AtomsPack.hpp"
#include "Ptensor1.hpp"
#include "Cgraph.hpp"

namespace ptens{


  class Ptensor1subpack: public cnine::RtensorA{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;

    AtomsPack atoms;


    // ---- Constructors -------------------------------------------------------------------------------------


    Ptensor1subpack(const int _k, const int _nc, const int _dev=0): 
      rtensor({0,_k,_nc},cnine::fill_raw(),_dev),
      atoms(_k,_dev){}


    Ptensor1subpack(const PtensorSgntr& sgntr, const int _dev=0):
      rtensor({0,sgntr.k,sgntr.nc},cnine::fill_raw(),_dev),
      atoms(sgntr.k,_dev){}


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    Ptensor1subpack(const AtomsPack& _atoms, const int nc, const FILLTYPE& dummy, const int _dev=0):
      rtensor(cnine::Gdims(_atoms.getn(),_atoms.getk(),nc),dummy,_dev),
      atoms(_atoms){}


  public: // ---- Constructors --------------------------------------------------------------------------------
    

    static Ptensor1subpack raw(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensor1subpack(_atoms,nc,cnine::fill_raw(),_dev);}

    static Ptensor1subpack zero(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensor1subpack(_atoms,nc,cnine::fill_zero(),_dev);}

    static Ptensor1subpack gaussian(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensor1subpack(_atoms,nc,cnine::fill_gaussian(),_dev);}

    static Ptensor1subpack sequential(const AtomsPack& _atoms, const int nc, const int _dev=0){
      return Ptensor1subpack(_atoms,nc,cnine::fill_sequential(),_dev);}

    
  public: // ---- Copying -------------------------------------------------------------------------------------


    Ptensor1subpack(const Ptensor1subpack& x):
      rtensor(x),
      atoms(x.atoms){
    }


    Ptensor1subpack(Ptensor1subpack&& x):
      rtensor(std::move(x)),
      atoms(std::move(x.atoms)){
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    Ptensor1subpack(const AtomsPack& _atoms, const rtensor& x):
      rtensor(x), 
      atoms(_atoms){
      assert(atoms.getn()==getn());
      assert(atoms.getk()==getk());
    }

   Ptensor1subpack(const AtomsPack& _atoms, rtensor&& x):
     rtensor(std::move(x)), 
     atoms(_atoms){
      assert(atoms.getn()==getn());
      assert(atoms.getk()==getk());
    }


  public: // ---- Access --------------------------------------------------------------------------------------
    

    int getn() const{
      return dims(0);
    }

    int getk() const{
      return dims(1);
    }

    int get_nc() const{
      return dims(2);
    }

    int push_back(const Ptensor1& x){
      assert(x.getk()==getk());
      assert(x.get_nc()==get_nc());
      push_back_slice0(x);
      atoms.push_back(x.atoms);
      return getn()-1;
    }

    Ptensor1 tensor(const int i){
      assert(i<getn());
      return Ptensor1(slice0(i),Atoms(atoms(i)));
    }

    Ptensor1 view_of_tensor(const int i){
      assert(i<getn());
      return Ptensor1(view_of_slice0(i),Atoms(atoms(i)));
    }

    const Ptensor1 view_of_tensor(const int i) const{
      assert(i<getn());
      return Ptensor1(const_cast<Ptensor1subpack&>(*this).view_of_slice0(i),Atoms(atoms(i)));
    }

    void forall_tensors(const std::function<void(Ptensor1)>& lambda){
      for(int i=0; i<getn(); i++)
	lambda(view_of_tensor(i));
    }

    void forall_tensors(const std::function<void(const Ptensor1)>& lambda) const{
      for(int i=0; i<getn(); i++)
	lambda(view_of_tensor(i));
    }


  public: // ---- Message passing -----------------------------------------------------------------------------


    void collect(const Ptensor1subpack& x, const Cgraph& graph){
      graph.forall_edges([&](const int i, const int j){
	  view_of_tensor(i).pull_msg(x.view_of_tensor(j));});
    }



  public: // ---- Reductions ----------------------------------------------------------------------------------


    rtensor reduce0(const 


  public: // ---- Operations ----------------------------------------------------------------------------------


    Ptensor1subpack operator*(const rtensor& W) const{
      assert(W.getk()==2);
      assert(W.dim(0)==get_nc());
      Ptensor1subpack R(atoms,W.dim(1),cnine::fill_zero(),dev);
      R.add_mprod(*this,W);
      return R;
    }

    
    void add_mprod(const Ptensor1subpack& x, const rtensor& W){
      add_mprod_AA(x,W);
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      forall_tensors([&](const Ptensor1& x){
	  oss<<x.str(indent)<<endl;});
      return oss.str();
    }
	  
    friend ostream& operator<<(ostream& stream, const Ptensor1subpack& x){
      stream<<x.str(); return stream;}
   


  };


}


#endif 
