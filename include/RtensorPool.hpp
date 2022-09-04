#ifndef _RtensorPool
#define _RtensorPool

#include "vector_pool.hpp"
#include "RtensorA.hpp"

namespace ptens{

  class RtensorPool{
  public:

    typedef cnine::RtensorA rtensor;

    float* arr=nullptr;
    float* arrg=nullptr;
    int dev=0;
    int memsize=0;
    int tail=0;
    vector_pool<int> headers;

    ~RtensorPool(){
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPool(){}


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n){
      if(n<=memsize) return;
      int newsize=n;
      if(dev==0){
	float* newarr=new float[newsize];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	float* newarrg;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return headers.size();
    }

    int addr_of(const int i) const{
      assert(i<size());
      return headers(i)[0];
    }

    cnine::Gdims dims_of(const int i) const{
      assert(i<size());
      return cnine::Gdims(headers.subvector_of(i,1));
    }

    float* get_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }


    rtensor operator()(const int i) const{
      assert(i<size());
      return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    }

    rtensor tensor(const int i) const{
      assert(i<size());
      return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    }

    void push_back(const rtensor& x){
      assert(x.dev==dev);
      if(tail+x.asize>memsize)
	reserve(std::max(2*memsize,tail+x.asize));
      if(dev==0){
	std::copy(x.arr,x.arr+x.asize,arr+tail);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arr+tail,x.arrg,x.asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      headers.push_back_cat(tail,x.dims);
      tail+=x.asize;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Tensor "<<i<<":"<<endl;
	oss<<(*this)(i).str(indent)<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const RtensorPool& v){
      stream<<v.str(); return stream;}

  };

}

#endif
