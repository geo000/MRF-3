
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

// standard lib (STL or boost)
#include<map>
#include<vector>
#include<string>
//#include "boost/algorithm/string.hpp"
//#include"boost\lexical_cast.hpp"

//standard lib(STL or boost)

#include<io.h>
#include<iostream>
#include<fstream>
#include<opencv2\opencv.hpp>
#include<sstream>
#include<ctime>
#include<assert.h>
#include<direct.h>


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

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 1024 threads per block
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
extern int CUDA_GET_BLOCKS(const int N); 

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
	typedef std::vector< std::vector<cv::Point> > PointsArrays;
	extern class GraphCut4MRF;
	extern class MyDataCostFunctor;
	extern class MySmoothCostFunctor;
/***********************************  graphcut definition ******************************/


/***********************************  gflags  setting     ******************************/


	DECLARE_string(infolder);
	DECLARE_string(gpu);
	DECLARE_string(imageName);
	DECLARE_bool(dumpImage);
	DECLARE_string(dumpName);
	DECLARE_bool(showEdge);
	DECLARE_string(edgeSolver);
	DECLARE_bool(showInitialImage);
	DECLARE_bool(dumpInitialImage);
	DECLARE_string(graphSolver);

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
				}							

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

namespace CUDA
{
	extern void get_gpus(std::vector<int>* gpus);



}



namespace TK
{
	extern  bool tk_is_file_existed(const char* filename);

	extern  bool tk_make_file(const char* filename);

	template<class T>
	extern bool tk_normalize(T** data, int Y, int X);


	template<class T>
	extern bool tk_check(T** data, int Y, int X);

	template<class T>
	extern bool tk_memset(T** data, int Y, int X);


	extern bool tk_save_img(const cv::Mat img, const std::string& filename );

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

	extern void tk_get_mat_array(const std::string & filename, MatArray& output);


}
