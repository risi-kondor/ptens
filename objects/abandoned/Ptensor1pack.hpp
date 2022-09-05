#ifndef _ptens_Ptensor1pack
#define _ptens_Ptensor1pack

#include "Cgraph.hpp"
#include "Ptensor1subpack.hpp"
#include "PtensorSubpackSpecializer.hpp"



namespace ptens{


  class Ptensor1pack{
  public:

    typedef cnine::IntTensor itensor;
    typedef cnine::RtensorA rtensor;

    mutable vector<Ptensor1subpack*> subpacks;
    mutable unordered_map<PtensorSgntr,int> sgntr_lookup; 
    mutable unordered_map<int,pair<int,int> > index_lookup;
    int max_index=0;

    ~Ptensor1pack(){
      for(auto p:subpacks)
	delete p;
    }


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensor1pack(){}


  public: // ----- Constructors ------------------------------------------------------------------------------


    Ptensor1pack(const Ptensor1pack& x){
      for(auto p:x.subpacks)
	subpacks.push_back(new Ptensor1subpack(*p)); 
      sgntr_lookup=x.sgntr_lookup;
      index_lookup=x.index_lookup;
      max_index=x.max_index;
    }
	
    Ptensor1pack(Ptensor1pack&& x){
      subpacks=x.subpacks;
      x.subpacks.clear();
      sgntr_lookup=x.sgntr_lookup;
      index_lookup=x.index_lookup;
      max_index=x.max_index;
    }

    Ptensor1pack& operator=(const Ptensor1pack& x)=delete;


  public: // ----- Access ------------------------------------------------------------------------------------


    pair<int,int> index_of_tensor(const int i) const{
      assert(index_lookup.find(i)!=index_lookup.end());
      return index_lookup[i];
    }

    int index_of_subpack(const PtensorSgntr& sgntr) const{
      auto it=sgntr_lookup.find(sgntr);
      if(it!=sgntr_lookup.end()) return it->second;
      subpacks.push_back(new Ptensor1subpack(sgntr));
      int i=subpacks.size()-1;
      sgntr_lookup[sgntr]=i;
      return i;
    }

    void insert(const int ix, const Ptensor1& x){
      assert(index_lookup.find(ix)==index_lookup.end());
      int pix=index_of_subpack(x.signature());
      int subix=subpacks[pix]->push_back(x);
      index_lookup[ix]=make_pair(pix,subix);
      max_index=std::max(max_index,ix);
    }

    int push_back(const Ptensor1& x){
      int pix=index_of_subpack(x.signature());
      int subix=subpacks[pix]->push_back(x);
      index_lookup[max_index]=make_pair(pix,subix);
      return max_index++;
    }


  public: // ---- Message passing ----------------------------------------------------------------------------


    void collect(const Ptensor1pack& x, const Cgraph& graph){
      PtensorSubgraphSpecializer execs;
      
      graph.forall_edges([&](const int i, const int j){
	  auto ix0=index_of_tensor(i);
	  auto ix1=x.index_of_tensor(j);
	  execs.graph(ix0.first,ix1.first).push(ix0.second,ix1.second);
	});

      execs.forall([&](const int i, const int j, const Cgraph& graph){
	  subpacks[i]->collect(*subpacks[j],graph);});
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    Ptensor1pack operator*(const rtensor& W){
      Ptensor1pack R;
      for(auto p:subpacks)
	R.subpacks.push_back(new Ptensor1subpack(p->atoms,(*p)*W));
      R.sgntr_lookup=sgntr_lookup;
      R.index_lookup=index_lookup;
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto p:subpacks)
	oss<<p->str(indent)<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Ptensor1pack& x){
      stream<<x.str(); return stream;}
   

  };


}


#endif 
