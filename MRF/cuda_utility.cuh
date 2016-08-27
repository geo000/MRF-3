
#include"Utility.h"

#include"thrust\host_vector.h"
#include"thrust\device_vector.h"

extern cudaEvent_t start;
extern cudaEvent_t end;

typedef struct device_point
{
	int x;
	int y;
}device_point;

namespace CUDA{
	 
	//cuda kernels' function 
	extern __global__ void kernel_get_points(const uchar** pointMask, thrust::device_vector<device_point>, const int pixel_num);

	//  wrapper
	extern __global__ void get_scribble_points(const MatArray& pointsMask, const MatArray& slicPointsMask, PointsArrays& points, PointsArrays& slicPoints);

	

}
