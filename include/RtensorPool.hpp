#ifndef _RtensorPool
#define _RtensorPool

#include "array_pool.hpp"
#include "vector_pool.hpp"
#include "RtensorA.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"


namespace ptens{

  class RtensorPool{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::RtensorA rtensor;
    typedef cnine::Rtensor1_view Rtensor1_view;
    typedef cnine::Rtensor2_view Rtensor2_view;
    typedef cnine::Rtensor3_view Rtensor3_view;

    float* arr=nullptr;
    float* arrg=nullptr;
    int dev=0;
    int memsize=0;
    int tail=0;
    vector_pool<int> headers;

    ~RtensorPool(){
      //if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPool(){}

    RtensorPool(const int _dev):
      dev(_dev){}

    RtensorPool(const int _N, const Gdims& _dims, const cnine::fill_raw& dummy, const int _dev=0):
      RtensorPool(_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      for(int i=0; i<_N; i++){
	headers.push_back_cat(i*asize,_dims);
      }
      tail=_N*asize;
    }

    RtensorPool(const int _N, const Gdims& _dims, const cnine::fill_zero& dummy, const int _dev=0):
      RtensorPool(_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1){}
      for(int i=0; i<_N; i++)
	headers.push_back_cat(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPool(const array_pool<int>& dimensions, const cnine::fill_zero& dummy, const int _dev=0){
      dev=_dev;

      int reserve_size=0;
      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	reserve_size+=t;
      }
      reserve(reserve_size);
      if(dev==0) std::fill(arr,arr+reserve_size,0);
      if(dev==1){}

      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	headers.push_back_cat(tail,v);
	tail+=t;
      }

    }


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
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


    void reserve_zero(const int n){
      if(n<=memsize) return;
      //int newsize=n;
      if(dev==0){
	float* newarr=new float[n];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	std::fill(arr+memsize,arr+n,0);
	memsize=n;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, n*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	CUDA_SAFE(cudaMemset(arrg+memsize,0,(n-memsize)*sizeof(float)));
	memsize=n;
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return headers.size();
    }

    float* get_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }


    int addr_of(const int i) const{
      assert(i<size());
      return headers(i)[0];
    }

    cnine::Gdims dims_of(const int i) const{
      assert(i<size());
      return cnine::Gdims(headers.subvector_of(i,1));
    }

    int dim_of(const int i, const int j) const{
      assert(i<size());
      return headers(i,1+j);
    }

    float* arr_of(const int i) const{
      if(dev==1) return arrg+addr_of(i);
      return arr+addr_of(i);
    }




    rtensor operator()(const int i) const{
      assert(i<size());
      return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    }

    //rtensor tensor(const int i) const{
    //assert(i<size());
    //return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    //}

    Rtensor1_view view1_of(const int i) const{
      vector<int> v=headers(i);
      assert(v.size()==2);
      if(dev==1) return Rtensor1_view(arrg+v[0],v[1],1,1);
      return Rtensor1_view(arr+v[0],v[1],1,0);
    }

    Rtensor2_view view2_of(const int i) const{
      vector<int> v=headers(i);
      assert(v.size()==3);
      if(dev==1) return Rtensor2_view(arrg+v[0],v[1],v[2],v[2],1,1);
      return Rtensor2_view(arr+v[0],v[1],v[2],v[2],1,0);
    }

    Rtensor3_view view3_of(const int i) const{
      vector<int> v=headers(i);
      assert(v.size()==4);
      if(dev==1) return Rtensor3_view(arrg+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,1);
      return Rtensor3_view(arr+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,0);
    }



    void push_back(const rtensor& x){
      assert(x.dev==dev);
      if(tail+x.asize>memsize)
	reserve(std::max(2*memsize,tail+x.asize));
      if(dev==0){
	std::copy(x.arr,x.arr+x.asize,arr+tail);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg+tail,x.arrg,x.asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      headers.push_back_cat(tail,x.dims);
      tail+=x.asize;
    }

    void push_back_raw(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      headers.push_back_cat(tail,_dims);
      tail+=asize;
    }
      
    void push_back_zero(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      if(dev==0){
	std::fill(arr+tail,arr+tail+asize,0);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg+tail,0,asize*sizeof(float)));
      }
      headers.push_back_cat(tail,_dims);
      tail+=asize;
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