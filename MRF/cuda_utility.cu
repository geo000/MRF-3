#include"cuda_utility.cuh"


namespace CUDA{


	// 
	__global__ void get_scribble_points(const float** pointMask, int*** & points, const int label_num, const int pixel_num, const int W, const int H){

		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		int offset = x + y * blockDim.x * gridDim.x;


	}


}