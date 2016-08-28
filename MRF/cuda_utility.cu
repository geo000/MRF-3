#include"cuda_utility.cuh"

 cudaEvent_t start_timer;
 cudaEvent_t end_timer;

//extern texture<ushort4, cudaTextureType3D, cudaReadModeElementType> device_textures3D;

#define CREATE_CUDA_TIMER						\
CUDA_CHECK(cudaEventCreate(&start_timer));		\
CUDA_CHECK(cudaEventCreate(&end_timer));		\
CUDA_CHECK(cudaEventRecord(start_timer,0));		\

#define END_CUDA_TIMER							\
CUDA_CHECK(cudaEventRecord(end_timer,0));		\
CUDA_CHECK(cudaEventSynchronize(stop));			\
float elapsedTime=0;							\
CUDA_CHECK(cudaEventElapsedTime(&elapsedTime,start_timer,end_timer));\
printf("Total time : %3.1f ms\n",elapsedTime);	\
CUDA_CHECK(cudaEventDestroy(start_timer));		\
CUDA_CHECK(cudaEventDestroy(end_timer));		\

namespace CUDA{

	
	__global__ void kernel_get_points(const uchar* pointMask, thrust::device_vector<device_point>& device_points_vec, const int pixel_num)
	{
		int offset = threadIdx.x + threadIdx.y *blockDim.x;

		if (offset >= pixel_num)  return;

		uchar ch = pointMask[offset];
		if (ch < MAX_VALUE){
			 


		}
		
	}
	// 
	void get_scribble_points(const MatArray& pointsMask, const MatArray& slicPointsMask, PointsArrays& points, PointsArrays& slicPoints){


		//get basic infos
		int label_num = pointsMask.size();
		
		CHECK_GT(label_num, 0) << "Label_num must greater to 0.\n";

		int width  = pointsMask[0].cols;	// width	->     cols   -   x
		int height = pointsMask[0].rows;	// height	->     rows   -   y





	}



}