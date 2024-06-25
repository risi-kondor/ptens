/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 */


#ifndef _PtensLoggedTimer
#define _PtensLoggedTimer

#include <fstream>
#include <chrono>
#include <ctime>

//#include "PtensSessionObj.hpp"

//extern ptens::PtensLog* ptens_log;
//extern ptens::PtensSessionObj* ptens::ptens_session;


namespace ptens{

  class LoggedTimer{
  public:

    string task;
    long long n_ops=0;
    chrono::time_point<chrono::system_clock> t0;

    ~LoggedTimer(){
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      //if(n_ops>0) ptens_session->log(task+" "+to_string(elapsed)+" ms"+" ["+to_string((int)(((float)n_ops)/elapsed/1000.0))+" Mflops]");
      //else ptens_session->log(task+" "+to_string(elapsed)+" ms");
    }


    LoggedTimer(string _task=""):
      task(_task){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ>
    LoggedTimer(string _task, const OBJ& obj):
      task(_task+obj.repr()){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ>
    LoggedTimer(string _task, const OBJ& obj, const string s1):
      task(_task+obj.repr()+s1){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ0, typename OBJ1>
    LoggedTimer(string _task, const OBJ0& obj0, const string s1, const OBJ1& obj1):
      task(_task+obj0.repr()+s1+obj1.repr()){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ0, typename OBJ1>
    LoggedTimer(string _task, const OBJ0& obj0, const string s1, const OBJ1& obj1, const string s2):
      task(_task+obj0.repr()+s1+obj1.repr()+s2){
      t0=chrono::system_clock::now();
    }


    template<typename OBJ0>
    LoggedTimer(const OBJ0& obj0, string _task):
      task(obj0.repr()+_task){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ0, typename OBJ1>
    LoggedTimer(const OBJ0& obj0, string _task, const OBJ1& obj1):
      task(obj0.repr()+_task+obj1.repr()){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ0, typename OBJ1>
    LoggedTimer(const OBJ0& obj0, string _task, const OBJ1& obj1, const string s1):
      task(obj0.repr()+_task+obj1.repr()+s1){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ0, typename OBJ1, typename OBJ2>
    LoggedTimer(const OBJ0& obj0, string _task, const OBJ1& obj1, const string s1, const OBJ2& obj2):
      task(obj0.repr()+_task+obj1.repr()+s1+obj2.repr()){
      t0=chrono::system_clock::now();
    }

    template<typename OBJ0, typename OBJ1, typename OBJ2>
    LoggedTimer(const OBJ0& obj0, string _task, const OBJ1& obj1, const string s1, const OBJ2& obj2, const string s2):
      task(obj0.repr()+_task+obj1.repr()+s1+obj2.repr()+s2){
      t0=chrono::system_clock::now();
    }




    LoggedTimer(string _task, const long long _ops):
      task(_task){
      t0=chrono::system_clock::now();
      n_ops=_ops;
    }


  };


  class TimedFn: public LoggedTimer{
  public:

    template<typename OBJ0>
    TimedFn(string cl, string fn, const OBJ0& obj0):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+")"){}

    template<typename OBJ0, typename OBJ1>
    TimedFn(string cl, string fn, const OBJ0& obj0, const OBJ1& obj1):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+","+obj1.repr()+")"){}

    template<typename OBJ0, typename OBJ1, typename OBJ2>
    TimedFn(string cl, string fn, const OBJ0& obj0, const OBJ1& obj1, const OBJ2& obj2):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+","+obj1.repr()+","+obj2.repr()+")"){}


    template<typename OBJ0>
    TimedFn(string cl, string fn, const OBJ0& obj0, const int count):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+")"+" [n="+to_string(count)+"]",(long long) count){}

    template<typename OBJ0, typename OBJ1>
    TimedFn(string cl, string fn, const OBJ0& obj0, const OBJ1& obj1, const int count):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+","+obj1.repr()+")"+" [n="+to_string(count)+"]",(long long) count){}

    template<typename OBJ0, typename OBJ1, typename OBJ2>
    TimedFn(string cl, string fn, const OBJ0& obj0, const OBJ1& obj1, const OBJ2& obj2, const int count):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+","+obj1.repr()+","+obj2.repr()+")"+" [n="+to_string(count)+"]",(long long) count){}


    template<typename OBJ0>
    TimedFn(string cl, string fn, const OBJ0& obj0, const long long count):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+")"+" [n="+to_string(count)+"]",count){}

    template<typename OBJ0, typename OBJ1>
    TimedFn(string cl, string fn, const OBJ0& obj0, const OBJ1& obj1, const long long count):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+","+obj1.repr()+")"+" [n="+to_string(count)+"]",count){}

    template<typename OBJ0, typename OBJ1, typename OBJ2>
    TimedFn(string cl, string fn, const OBJ0& obj0, const OBJ1& obj1, const OBJ2& obj2, const long long count):
      LoggedTimer(cl+"::"+fn+"("+obj0.repr()+","+obj1.repr()+","+obj2.repr()+")"+" [n="+to_string(count)+"]",count){}


  };


  class TimedBlock: public LoggedTimer{
  public:

    TimedBlock(string _name, std::function<void()> lambda):
      LoggedTimer(_name){
      lambda();
    }

  };

}

#endif 
