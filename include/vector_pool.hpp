#ifndef _vector_pool
#define _vector_pool

#include "Ptens_base.hpp"

namespace ptens{

  template<typename TYPE>
  class vector_pool{
  public:

    TYPE* arr=nullptr;
    TYPE* arrg=nullptr;
    int memsize=0;
    int tail=0;
    int dev=0;

    vector<pair<int,int> > lookup;

    ~vector_pool(){
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    vector_pool(){}


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


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return lookup.size();
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

    vector<TYPE> subvector_of(const int i, const int beg) const{
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

    void push_back(const vector<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      for(int i=0; i<len; i++)
	arr[tail+i]=v[i];
      lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }

    void push_back(const initializer_list<TYPE>& v){
      push_back(vector<TYPE>(v));
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


    friend ostream& operator<<(ostream& stream, const vector_pool& v){
      stream<<v.str(); return stream;}

  };

}

#endif
