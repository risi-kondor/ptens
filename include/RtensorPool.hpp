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
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    //RtensorPool(){}

    RtensorPool(const int _dev=0):
      dev(_dev){}

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
