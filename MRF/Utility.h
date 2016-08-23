
/* Utility.h */
/*
This software library implements some common useful algorithm

This algorithm was developed by Lechao Cheng(liygcheng@zju.edu.cn)
at Zhejiang university. 

It's free for you to use this software for research purposes except 
for commercial usage.

----------------------------------------------------------------------

REUSING TREES:

If you use this option, you should cite
the aforementioned paper in any resulting publication.
*/





#pragma once


#include<map>
#include<vector>


#include<io.h>
#include<iostream>
#include<fstream>
#include<opencv2\opencv.hpp>
#include<sstream>
#include<ctime>
#include<assert.h>


//cuda header
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
//cuda header 

//
#include<gflags\gflags.h>
#include<glog\logging.h>

/***********************************  cuda macros **************************************/
#define CUDA_CHECK(condition) \
	/*code block avoids redefinition of cudaError_t error */	\
	cudaError_t error = condition;								\
	CHECK_EQ(error,cudaSuccess)<<""<<cudaGetErrorString(error); \

#define CUBLAS_CHECK(condition)\
		cublasStatus_t status = condition;\
		CHECK_EQ(status,CUBLAS_STATUS_SUCCESS) <<" " \
		<< TK::cublasGetErrorString(status);\

#define CURAND_CHECK(condition)\
	curandStatus_t status =  condition;\
	CHECK_EQ(status,CURAND_STATUS_SUCCESS)<<" "\
	<<TK::curandGetErrorString(status);\

/***********************************  cuda macros **************************************/


/***********************************  some useful macros  ******************************/
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

#define MAX_VALUE 255
#define INF_VALUE 10000000

/***********************************  some useful macros  ******************************/



/***********************************  graphcut definition ******************************/
	typedef std::vector<cv::Mat>  MatArray;
	typedef std::vector<std::string> StrArray;
	extern class GraphCut4MRF;
	extern class MyDataCostFunctor;
	extern class MySmoothCostFunctor;
/***********************************  graphcut definition ******************************/


/***********************************  gflags  setting     ******************************/
	typedef int(*RegisterFunction)(void);
	typedef std::map<std::string, RegisterFunction> RegisterFunMap;
	extern RegisterFunMap fun_map;

	#define doRegisteration(fun)	\
	namespace{						\
	class _Register_##fun{			\
	public:	_Register_##fun()		\
		{							\
	fun_map[#fun] = &(fun);			\
		}							\
	};							\
	_Register_##fun m_registeration_##fun;\
	}							\


	static RegisterFunction  getCommandFunction(const std::string& name)
	{
		if (fun_map.count(name)){
			return fun_map[name];
		}
		else
		{
			LOG(ERROR) << "Available Actions:";
			for (RegisterFunMap::iterator it = fun_map.begin(); it != fun_map.end(); ++it)
			{
				LOG(ERROR) << "\t" << it->first;
			}
			LOG(FATAL) << "unknown actions :" << name;
			return NULL;

		}

	}

/***********************************  gflags  setting     ******************************/




namespace TK
{
	extern  bool tk_is_file_existed(const char* filename);

	template<class T>
	extern bool tk_normalize(T** data, int Y, int X);


	template<class T>
	extern bool tk_check(T** data, int Y, int X);

	template<class T>
	extern bool tk_memset(T** data, int Y, int X);


	extern bool tk_save_img(const cv::Mat img, const char* filename = "./subtotal/result.png");

	template<class T>
	extern bool tk_dump_vec(const std::vector<std::vector<T>> data, int Y, int X, const char* filename);

	template<class T>
	extern bool tk_elicit_vec(std::vector<std::vector<T>> & data, int Y, int X, const char* filename);



	extern bool tk_dump_points(const std::vector<std::vector<cv::Point>> data, const char* filename);



	extern bool tk_elicit_points(std::vector<std::vector<cv::Point>> & data, const char* filename);

	template<class T>
	extern bool tk_dump_malloc(T** data, int Y, int X, const char* filename);

	template<class T>
	extern bool tk_elicit_malloc(T** & data, int Y, int X, const char* filename);


	template<class T>
	extern std::string tk_toString(const T& t);

	extern const char* cublasGetErrorString(cublasStatus_t error);
	extern const char* curandGetErrorString(curandStatus_t error);


}
