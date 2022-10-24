#ifndef _ptens_AindexPack
#define _ptens_AindexPack

#include <map>

#include "array_pool.hpp"
#include "Atoms.hpp"
#include "GatherMap.hpp"


namespace ptens{


  class AindexPack: public cnine::array_pool<int>{
  public:

    cnine::GatherMap const* bmap=nullptr; // hack


  public: // ---- Constructors ------------------------------------------------------------------------------


    AindexPack(){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    AindexPack(const AindexPack& x):
      array_pool<int>(x){
      bmap=x.bmap;}

    AindexPack(AindexPack&& x):
      array_pool<int>(std::move(x)){
      bmap=x.bmap; x.bmap=nullptr;
    }

    AindexPack& operator=(const AindexPack& x)=delete;


  public: // ---- Access -------------------------------------------------------------------------------------


    int tix(const int i) const{
      assert(i<size());
      //return arr[lookup[i].first];
      return arr[dir(i,0)];
    }

    int tens(const int i) const{
      assert(i<size());
      return arr[dir(i,0)];
      //return arr[lookup[i].first];
    }

    vector<int> ix(const int i) const{
      assert(i<size());
      int addr=dir(i,0);
      int len=dir(i,1)-1;
      PTENS_ASSRT(len>=0);
      vector<int> R(len);
      for(int i=0; i<len; i++){
	R[i]=arr[addr+i+1];
      }
      return R;
    }

    int ix(const int i, const int j) const{
      assert(i<size());
      int addr=dir(i,0);
      int len=dir(i,1);
      assert(len>=0);
      return arr[addr+j];
    }

    int nix(const int i) const{
      assert(i<size());
      return dir(i,1)-1;
    }

    /*
    int nindices(const int i) const{
      assert(i<size());
      //return lookup[i].second-1;
      return dir(i,1)-1;
    }
    */

    const cnine::GatherMap& get_bmap() const{
      assert(bmap);
      return *bmap;
    }

    int* get_barr(const int _dev=0) const{
      assert(bmap);
      const_cast<cnine::GatherMap*>(bmap)->to_device(_dev);
      if(_dev==0) return bmap->arr;
      return bmap->arrg;
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
    /*
    vector<int> indices(const int i) const{ // ????
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
    */

