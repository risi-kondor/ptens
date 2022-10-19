#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "LinmapFunctions.hpp"
#include "MsgFunctions.hpp"

using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;

  //auto A=Ptensor2::gaussian({0,1,2,3},3);
  auto A=Ptensor2::sequential({0,1,2,3},1);
  cout<<A<<endl;
  cout<<linmaps0(A)<<endl;
  cout<<linmaps1(A)<<endl;
  cout<<linmaps2(A)<<endl;
  cout<<"---------------"<<endl<<endl;

  cout<<A<<endl;
  auto B=Ptensor1::zero({0,1,5},5);
  A>>B; //add_msg(B,A);
  cout<<B<<endl;

  auto C=Ptensor2::zero({0,1,5},15);
  A>>C; //add_msg(C,A);
  cout<<C<<endl;



}
