#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "AtomsPack.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  Atoms a({1,2,3,4,5});

  AtomsPack atoms=AtomsPack::sequential(10,5);

  atoms.push_back(a);

  cout<<atoms<<endl;

}
