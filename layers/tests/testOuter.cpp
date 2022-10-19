#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "OuterLayers.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensors0 x0=Ptensors0::sequential(3,2);
  Ptensors0 y0=Ptensors0::sequential(3,2);

  cout<<outer(x0,y0)<<endl;
  

}
