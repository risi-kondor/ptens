#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "RtensorPool.hpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  RtensorPool pool;

  pool.push_back(RtensorA::sequential({3,3}));
  pool.push_back(RtensorA::sequential({4,4}));

  cout<<pool<<endl;

}
