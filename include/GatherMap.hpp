#ifndef _GatherMap
#define _GatherMap

#include "Ptens_base.hpp"
#include "array_pool.hpp"

namespace ptens{


  class GatherList{
  public:

    float* arr=nullptr;
    int n=0;
    bool is_view=false;

    ~GatherList(){
      if(is_view) return;
      delete[] arr;
    }

  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherList(float* _arr, const int _n):
      arr(_arr), n(_n), is_view(true){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    GatherList(const GatherList& x){
      n=x.n;
      arr=new float[2*n];
      std::copy(x.arr,x.arr+2*n,arr);
    }

    GatherList(GatherList&& x){
      n=x.n; x.n=0;
      arr=x.arr; x.arr=nullptr;
      is_view=x.is_view;
    }

    GatherList& operator=(const GatherList& x)=delete;
    

  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return n;
    }

    pair<int,float> operator()(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherList::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return pair<int,float>(arr[2*i],arr[2*i+1]);
    }

    int src(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherList::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return arr[2*i];
    }

    int weight(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherList::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return arr[2*i+1];
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<n; i++){
	oss<<"("<<static_cast<int>(arr[2*i])<<","<<arr[2*i+1]<<")";
	if(i<n-1) oss<<",";
      }
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const GatherList& v){
      stream<<v.str(); return stream;}

  };



  class GatherMap: public array_pool<float>{
  public:

  public: // ---- Copying ------------------------------------------------------------------------------------

    GatherMap(const GatherMap& x):
      array_pool<float>(x){}

    GatherMap(GatherMap&& x):
      array_pool<float>(std::move(x)){}

    GatherMap& operator=(const GatherMap& x){
      array_pool<float>::operator=(x);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    const GatherList operator()(const int i) const{
      CNINE_CPUONLY();
      CNINE_CHECK_RANGE(if(i>=size()) throw std::out_of_range("In GatherMap::operator(): index "+to_string(i)+" out of range (0,"+to_string(size()-1)+")."));
      return GatherList(arr+dir(i,0),dir(i,1)/2);
    }

    forall(std::function<void(const int, const GatherList)> lambda) const{
      for(int i=0; i<n; i++)
	lambda(i,(*this)(i));
    }

    void push_back(const vector<int>& v){
      int len=2*v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      for(int i=0; i<len/2; i++){
	arr[tail+2*i]=v[i];
	arr[tail+2*i+1]=1.0;
      }
      dir.push_back(tail,len);
      tail+=len;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      forall([&](const int i, const GatherList lst){oss<<indent<<i<<": "<<lst<<endl;}
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMap& v){
      stream<<v.str(); return stream;}

  };

}
