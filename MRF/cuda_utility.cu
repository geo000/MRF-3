#ifndef _CUDA_UTILITY_CU
#define _CUDA_UTILITY_CU

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



	 //__global__ void kernel_get_points(uchar* data, device_point*  device_points, const int pixel_num, const int width)
	 //{
		// int offset = threadIdx.x + blockIdx.x * blockDim.x;

		// if (offset >= pixel_num)  return;
		// uchar ch = data[offset];
		// if (ch < MAX_VALUE)
		// {
		//	 int y = (int)(offset / width);
		//	 int x = offset%width;

		//	 device_point p(x,y);

		//	 //device_slic_vec.push_back(p);
		// }

	 //}
	 // 
	 //void get_scribble_points(const MatArray& pointsMask, const MatArray& slicPointsMask, PointsArrays& points, PointsArrays& slicPoints){


		// //get basic infos
		// int label_num = pointsMask.size();

		// CHECK_GT(label_num, 0) << "Label_num must greater to 0.\n";

		// int width = pointsMask[0].cols;	// width	->     cols   -   x
		// int height = pointsMask[0].rows;	// height	->     rows   -   y
		// int pixel_num = width * height;//

		// // device data pointers
		// uchar* device_masks;
		// uchar* device_slic_masks;
		// device_point*  device_points;
		// device_point*  device_slic_points;

		// // cudaMalloc
		// CUDA_CHECK(cudaMalloc((void**)&device_masks, sizeof(uchar) * pixel_num));
		// CUDA_CHECK(cudaMalloc((void**)&device_slic_masks, sizeof(uchar) * pixel_num));

		// CUDA_CHECK(cudaMalloc((void**)&device_points,sizeof(device_point) * pixel_num));
		// CUDA_CHECK(cudaMalloc((void**)&device_slic_points, sizeof(device_point) * pixel_num));



		// for (size_t i = 0; i < label_num; ++i)
		// {
		//	 //data clear
		//	 CUDA_CHECK(cudaMemset((void**)&device_points, 0, sizeof(device_point) * pixel_num));
		//	 CUDA_CHECK(cudaMemset((void**)&device_slic_points, 0, sizeof(device_point) * pixel_num));

		//	 //step 1: copy data to device
		//	 CUDA_CHECK(cudaMemcpy(device_masks, (void*)pointsMask[i].data, sizeof(uchar) * pixel_num, cudaMemcpyHostToDevice));
		//	 CUDA_CHECK(cudaMemcpy(device_slic_masks, (void*)slicPointsMask[i].data, sizeof(uchar) * pixel_num, cudaMemcpyHostToDevice));

		//	 //step 2: execute kernel
		//	// CUDA::kernel_get_points << <CUDA_GET_BLOCKS(pixel_num), CUDA_NUM_THREADS >> >(device_masks, device_points_result, pixel_num, width);

		//	// CUDA::kernel_get_points << <CUDA_GET_BLOCKS(pixel_num), CUDA_NUM_THREADS >> >(device_slic_masks, device_slic_result, pixel_num, width);

		//	 //step 3: get data back
		//	 //std::vector<device_point> temp_points(device_points_result.begin(), device_points_result.end());
		//	 //std::vector<device_point> temp_slic_points(device_slic_result.begin(), device_slic_result.end());

		//	 //points.push_back(temp_points);
		//	// slicPoints.push_back(temp_slic_points);
		// }





	 //}

 }
#endif
