#ifndef _CSRmatrix
#define _CSRmatrix

#include "Cnine_base.hpp"
#include "array_pool.hpp"

namespace cnine{


  template<typename TYPE>
  class svec{
  public:

    TYPE* arr=nullptr;
    int n=0;
    bool is_view=false;

    ~svec<TYPE>(){
      if(is_view) return;
      delete[] arr;
    }

  public: // ---- Constructors -------------------------------------------------------------------------------

    
    svec(const int _n): n(_n){
      arr=new int[2*n];
    }
    svec(const int _n, cnine::fill_zero& dummy): n(_n){
      arr=new int[2*n];
      for(int i=0; i<n; i++){
	arr[2*i]=reinterpret_cast<TYPE>(&i);
	arr[2*i+1]=0;
      }
    }

    svec(float* _arr, const int _n):
      arr(_arr), n(_n), is_view(true){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    svec(const svec<TYPE>& x){
      n=x.n;
      arr=new TYPE[2*n];
      std::copy(x.arr,x.arr+2*n,arr);
    }

    svec(svec<TYPE>&& x){
      n=x.n; x.n=0;
      arr=x.arr; x.arr=nullptr;
      is_view=x.is_view;
    }

    svec<TYPE>& operator=(const svec<TYPE>& x)=delete;
    

  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return n;
    }

    pair<int,float> operator[](const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In svec<TYPE>::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return pair<int,float>(reinterpret_cast<int>(arr[2*i]),arr[2*i+1]);
    }

    int ix(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In svec<TYPE>::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return reinterpret_cast<int>(arr[2*i]);
    }

    int val(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In svec<TYPE>::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return arr[2*i+1];
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<n; i++){
	oss<<"("<<*reinterpret_cast<int*>(arr+2*i)<<","<<arr[2*i+1]<<")";
	if(i<n-1) oss<<",";
      }
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const svec<TYPE>& v){
      stream<<v.str(); return stream;}

  };


  template<class TYPE>
  class CSRmatrix: public array_pool<TYPE>{
  public:

    using array_pool<TYPE>::arr;
    using array_pool<TYPE>::arrg;
    using array_pool<TYPE>::tail;
    using array_pool<TYPE>::memsize;
    using array_pool<TYPE>::dev;
    using array_pool<TYPE>::dir;

    using array_pool<TYPE>::size;


    CSRmatrix():
      array_pool<TYPE>(){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    CSRmatrix(const CSRmatrix<TYPE>& x):
      array_pool<TYPE>(x){}

    CSRmatrix(CSRmatrix<TYPE>&& x):
      array_pool<TYPE>(std::move(x)){}

    CSRmatrix& operator=(const CSRmatrix<TYPE>& x){
      array_pool<TYPE>::operator=(x);
    }

  
  public: // ---- Access -------------------------------------------------------------------------------------


    const svec<TYPE> operator()(const int i) const{
      CNINE_CPUONLY();
      CNINE_CHECK_RANGE(if(i>=size()) throw std::out_of_range("In CSRmatrix::operator(): index "+to_string(i)+" out of range (0,"+to_string(size()-1)+")."));
      return svec<TYPE>(arr+dir(i,0),dir(i,1)/2);
    }

    void for_each(std::function<void(const int, const svec<TYPE>)> lambda) const{
      for(int i=0; i<size(); i++)
	lambda(i,(*this)(i));
    }

    void push_back(const vector<int>& ix, const vector<TYPE>& v){
      int len=ix.size();
      PTENS_ASSRT(v.size()==len);
      if(tail+2*len>memsize)
	reserve(std::max(2*memsize,tail+2*len));
      for(int i=0; i<len; i++){
	arr[tail+2*i]=ix[i];
	arr[tail+2*i+1]=v[i];
      }
      dir.push_back(tail,2*len);
      tail+=2*len;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for_each([&](const int i, const svec<TYPE> lst){oss<<indent<<i<<": "<<lst<<endl;});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CSRmatrix& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
