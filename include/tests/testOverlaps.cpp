#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Hgraph.hpp"
#include "AtomsPack.hpp"

using namespace ptens;
using namespace cnine;

int main(int argc, char** argv){

  cnine_session session;

  AtomsPack x({{0,1},{1,2,3},{5}});
  AtomsPack y({{0},{1,2,3},{4,5},{6}});

  Hgraph G=Hgraph::overlaps(x,y);
  cout<<G.dense()<<endl;

}
