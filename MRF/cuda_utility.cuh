
#include"Utility.h"

#include"thrust\host_vector.h"
#include"thrust\device_vector.h"


namespace CUDA{

	extern __global__ void get_scribble_points(const float** pointMask,int*** & points,const int label_num,const int pixel_num,const int W,const int H);



}
