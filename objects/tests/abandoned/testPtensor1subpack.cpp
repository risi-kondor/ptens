#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ptensor1subpack.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  AtomsPack atoms=AtomsPack::sequential(3,5);
  Ptensor1subpack Ppack=Ptensor1subpack::sequential(atoms,5);

  Ptensor1 P1=Ptensor1::zero(Atoms::sequential(5),5);
  Ppack.push_back(P1);

  cout<<Ppack<<endl;

}
