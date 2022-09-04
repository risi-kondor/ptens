#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "vector_pool.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  vector_pool<int> pool;

  pool.push_back({3,5,7});
  pool.push_back({0,0,2,9});
  pool.push_back({1,4});

  cout<<pool<<endl;

}
