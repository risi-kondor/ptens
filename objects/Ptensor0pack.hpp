#ifndef _ptens_Ptensor0pack
#define _ptens_Ptensor0pack

#include "Cgraph.hpp"
#include "RtensorPool.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "Ptensor0.hpp"


namespace ptens{


  class Ptensor0pack: public RtensorPool{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    int nc;
    AtomsPack atoms;

    ~Ptensor0pack(){
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensor0pack(){}

    static Ptensor0pack zero(const AtomsPack& _atoms, const int _nc){
      Ptensor0pack R;
      R.nc=_nc;
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor0::zero(_atoms(i),_nc));
      }
      return R;
    }

    static Ptensor0pack sequential(const AtomsPack& _atoms, const int _nc){
      Ptensor0pack R;
      R.nc=_nc;
      for(int i=0; i<_atoms.size(); i++){
	R.push_back(Ptensor0::zero(_atoms(i),_nc));
	auto A=R.view1_of(i);
	for(int j=0; j<_nc; j++) A.set(j,i);
      }
      return R;
    }


  public: // ----- Copying -----------------------------------------------------------------------------------


    Ptensor0pack(const Ptensor0pack& x):
      RtensorPool(x),
      atoms(x.atoms){}
	
    Ptensor0pack(Ptensor0pack&& x):
      RtensorPool(std::move(x)),
      atoms(std::move(x.atoms)){}

    Ptensor0pack& operator=(const Ptensor0pack& x)=delete;


  public: // ----- Access ------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }

    Atoms atoms_of(const int i) const{
      return Atoms(atoms(i));
    }

    int push_back(const Ptensor0& x){
      if(size()==0) nc=x.get_nc();
      else assert(nc==x.get_nc());
      RtensorPool::push_back(x);
      atoms.push_back(x.atoms);
      return size()-1;
    }


  public: // ---- Message passing ----------------------------------------------------------------------------



  public: // ---- Reductions ---------------------------------------------------------------------------------


    RtensorPool messages0(const AindexPack& src_list) const{
      int N=src_list.size();

      array_pool<int> msg_dims;
      for(int i=0; i<N; i++)
	msg_dims.push_back({dims_of(src_list.tix(i)).back()});
      RtensorPool R(msg_dims,cnine::fill_zero());

      for(int i=0; i<N; i++)
	R.view1_of(i)=view1_of(src_list.tix(i));

      return R;
    }


  public: // ---- Broadcasting -------------------------------------------------------------------------------


    void add_messages0(const RtensorPool& messages, const AindexPack& dest_list, const int coffs){
      int N=dest_list.size();
      assert(messages.size()==N);

      for(int i=0; i<N; i++)
	view1_of(dest_list.tix(i))=messages.view1_of(i);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Ptensor "<<i<<" "<<Atoms(atoms(i))<<":"<<endl;
	oss<<RtensorPool::operator()(i).str()<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor0pack& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
