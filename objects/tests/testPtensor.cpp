
#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Atoms atoms({1,2,3});

  Ptensor A=Ptensor::zero(2,atoms);

  cout<<A<<endl;

}
