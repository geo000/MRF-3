
#include"Utility.h"

#include"thrust\host_vector.h"
#include"thrust\device_vector.h"


#include "texture_fetch_functions.h"
#include "texture_types.h"
#include "cuda_texture_types.h"

extern cudaEvent_t start_timer;
extern cudaEvent_t end_timer;


extern  texture<ushort4, cudaTextureType1D, cudaReadModeElementType> device_textures[4];


typedef struct device_point
{
	int x;
	int y;
}device_point;

namespace CUDA{
	 
	//cuda kernels' function 
	extern __global__ void kernel_get_points(const uchar* pointMask, thrust::device_vector<device_point>& device_points_vec, const int pixel_num);
	 
	//  wrapper
	extern  void get_scribble_points(const MatArray& pointsMask, const MatArray& slicPointsMask, PointsArrays& points, PointsArrays& slicPoints);

	

}
