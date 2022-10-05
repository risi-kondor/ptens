#ifndef _ptens_AindexPack
#define _ptens_AindexPack

#include <map>

#include "array_pool.hpp"
#include "Atoms.hpp"


namespace ptens{

  class AindexPack: public array_pool<int>{
  public:


  public: // ---- Constructors ------------------------------------------------------------------------------


    AindexPack(){}


  public: // ---- Constructors ------------------------------------------------------------------------------


  public: // ---- Access -------------------------------------------------------------------------------------


    int tix(const int i) const{
      assert(i<size());
      //return arr[lookup[i].first];
      return arr[dir(i,0)];
    }

    vector<int> indices(const int i) const{
      assert(i<size());
      //auto& p=lookup[i];
      //int addr=p.first+1;
      //int len=p.second-1;
      int addr=dir(i,0);
      int len=dir(i,1);
      assert(len>=0);
      vector<int> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }

    int tens(const int i) const{
      assert(i<size());
      return arr[dir(i,0)];
      //return arr[lookup[i].first];
    }

    vector<int> ix(const int i) const{
      assert(i<size());
      //auto& p=lookup[i];
      //int addr=p.first+1;
      //int len=p.second-1;
      int addr=dir(i,0);
      int len=dir(i,1);
      assert(len>=0);
      vector<int> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }

    int ix(const int i, const int j) const{
      assert(i<size());
      //auto& p=lookup[i];
      //int addr=p.first+1;
      //int len=p.second-1;
      int addr=dir(i,0);
      int len=dir(i,1);
      assert(len>=0);
      return arr[addr+j];
    }

    int nix(const int i) const{
      assert(i<size());
      //return lookup[i].second-1;
      return dir(i,1)-1;
    }

    int nindices(const int i) const{
      assert(i<size());
      //return lookup[i].second-1;
      return dir(i,1)-1;
    }

    void push_back(const int tix, vector<int> indices){
      int len=indices.size()+1;
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      arr[tail]=tix;
      for(int i=0; i<len-1; i++)
	arr[tail+1+i]=indices[i];
      dir.push_back(tail,len);
      //lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }

    
  public: // ---- Operations ---------------------------------------------------------------------------------

  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    string str(const string indent="") const{
      ostringstream oss;
      oss<<"(";
      for(int i=0; i<size()-1; i++)
	oss<<(*this)[i]<<",";
      if(size()>0) oss<<(*this)[size()-1];
      oss<<")";
      return oss.str();
    }
    */


    friend ostream& operator<<(ostream& stream, const AindexPack& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
