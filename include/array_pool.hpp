#ifndef _array_pool
#define _array_pool

#include "Ptens_base.hpp"

namespace ptens{

  template<typename TYPE>
  class array_pool{
  public:

    TYPE* arr=nullptr;
    TYPE* arrg=nullptr;
    int memsize=0;
    int tail=0;
    int dev=0;

    vector<pair<int,int> > lookup;

    ~array_pool(){
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    array_pool(){}


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n){
      if(n<=memsize) return;
      int newsize=n;
      if(dev==0){
	TYPE* newarr=new TYPE[newsize];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	TYPE* newarrg;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(TYPE)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    array_pool(const array_pool& x){
      dev=x.dev;
      tail=x.tail;
      memsize=tail;
      if(dev==0){
	arr=new TYPE[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_UNIMPL();
      }
      lookup=x.lookup;
    }

    array_pool(array_pool&& x){
      dev=x.dev;
      tail=x.tail; x.tail=0;
      memsize=x.memsize; x.memsize=0; 
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      lookup=std::move(x.lookup);
      x.lookup.clear();
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return lookup.size();
    }

    int size_of(const int i) const{
      assert(i<size());
      return lookup[i].second;
    }

    vector<TYPE> operator()(const int i) const{
      assert(i<size());
      auto& p=lookup[i];
      int addr=p.first;
      int len=p.second;
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }

    void push_back(const vector<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      for(int i=0; i<len; i++)
	arr[tail+i]=v[i];
      lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }

    void push_back(const set<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      int i=0; 
      for(TYPE p:v){
	arr[tail+i]=p;
	i++;
      }
      lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }

    void push_back(const initializer_list<TYPE>& v){
      push_back(vector<TYPE>(v));
    }

    void forall(const std::function<void(const vector<TYPE>&)>& lambda){
      int n=size();
      for(int i=0; i<n; i++)
	lambda((*this)(i));
    }


  public: // ---- Specialized --------------------------------------------------------------------------------

    /*
    vector<TYPE> subarray_of(const int i, const int beg) const{
      assert(i<size());
      auto& p=lookup[i];
      int addr=p.first+beg;
      int len=p.second-beg;
      assert(len>=0);
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }


    void push_back_cat(TYPE first, const vector<TYPE>& v){
      int len=v.size()+1;
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      arr[tail]=first;
      for(int i=0; i<len-1; i++)
	arr[tail+1+i]=v[i];
      lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	auto v=(*this)(i);
	oss<<"(";
	for(int j=0; j<v.size()-1; j++)
	  oss<<v[j]<<",";
	if(v.size()>0) oss<<v.back();
	oss<<")"<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const array_pool& v){
      stream<<v.str(); return stream;}

  };

}

#endif
