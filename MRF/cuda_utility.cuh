#ifndef _CUDA_UTILITY_H
#define _CUDA_UTILITY_H

#include"Utility.h"

#include"thrust\host_vector.h"
#include"thrust\device_vector.h"


#include "texture_fetch_functions.h"
#include "texture_types.h"
#include "cuda_texture_types.h"

#define  MAX_THREAD_X 1024
#define  MAX_THREAD_Y 1024
#define  MAX_THREAD_Z 64

extern cudaEvent_t start_timer;
extern cudaEvent_t end_timer;


//extern  texture<ushort4, cudaTextureType1D, cudaReadModeElementType> device_textures[2];



namespace CUDA{
	

	//cuda kernels' function 
	//extern __global__ void kernel_get_points(uchar* data, thrust::device_vector<device_point>& device_slic_vec, const int pixel_num, const int width);

	//  wrapper
	//extern   void get_scribble_points(const MatArray& pointsMask, const MatArray& slicPointsMask, PointsArrays& points, PointsArrays& slicPoints);


}

#endif
