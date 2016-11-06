#ifndef _CUDA_UTILITY_H
#define _CUDA_UTILITY_H

#include"Utility.h"

//#include"thrust\host_vector.h"
//#include"thrust\device_vector.h"


//#include "texture_fetch_functions.h"
//#include "texture_types.h"
//#include "cuda_texture_types.h"

#define  MAX_THREAD_X 1024
#define  MAX_THREAD_Y 1024
#define  MAX_THREAD_Z 64

extern cudaEvent_t start_timer;
extern cudaEvent_t end_timer;


//extern  texture<ushort4, cudaTextureType1D, cudaReadModeElementType> device_textures[2];

// 
#ifdef WIN32
#include <intrin.h>
#endif

#define safeCall(err)       __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync()    __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err) {
		fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
}

inline void __safeThreadSync(const char *file, const int line)
{
	cudaError err = cudaThreadSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
		exit(-1);
	}
}

inline bool deviceInit(int dev)
{
	int deviceCount;
	safeCall(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		return false;
	}
	if (dev < 0) dev = 0;
	if (dev > deviceCount - 1) dev = deviceCount - 1;
	cudaDeviceProp deviceProp;
	safeCall(cudaGetDeviceProperties(&deviceProp, dev));
	if (deviceProp.major < 1) {
		fprintf(stderr, "error: device does not support CUDA.\n");
		return false;
	}
	safeCall(cudaSetDevice(dev));
	return true;
}

class TimerGPU {
public:
	cudaEvent_t start, stop;
	cudaStream_t stream;
	TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}
	~TimerGPU() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	float read() {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
};

class TimerCPU
{
	static const int bits = 10;
public:
	long long beg_clock;
	float freq;
	TimerCPU(float freq_) : freq(freq_) {   // freq = clock frequency in MHz
		beg_clock = getTSC(bits);
	}
	long long getTSC(int bits) {
#ifdef WIN32
		return __rdtsc() / (1LL << bits);
#else
		unsigned int low, high;
		__asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
		return ((long long)high << (32 - bits)) | ((long long)low >> bits);
#endif
	}
	float read() {
		long long end_clock = getTSC(bits);
		long long Kcycles = end_clock - beg_clock;
		float time = (float)(1 << bits)*Kcycles / freq / 1e3f;
		return time;
	}
};





namespace CUDA{
	

	//cuda kernels' function 
	//extern __global__ void kernel_get_points(uchar* data, thrust::device_vector<device_point>& device_slic_vec, const int pixel_num, const int width);

	//  wrapper
	//extern   void get_scribble_points(const MatArray& pointsMask, const MatArray& slicPointsMask, PointsArrays& points, PointsArrays& slicPoints);


}

#endif
