#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor1pack.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Ptensor1pack Ppack;

  Ppack.push_back(Ptensor1::zero(Atoms::sequential(3),5));
  Ppack.push_back(Ptensor1::zero(Atoms::sequential(3),5));
  Ppack.push_back(Ptensor1::zero(Atoms::sequential(4),5));

  cout<<Ppack<<endl;

}
