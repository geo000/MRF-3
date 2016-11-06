#ifndef _CUDA_SIFT_CUH_
#define _CUDA_SIFT_CUH_

#include"cuda_utility.cuh"

//********************************Part 1  some constant var************************************* // cudaSiftD.h
#define NUM_SCALES      5

// Scale down thread block width
#define SCALEDOWN_W   160 

// Scale down thread block height
#define SCALEDOWN_H    16

// Find point thread block width
#define MINMAX_W      126 

// Find point thread block height
#define MINMAX_H        4

// Laplace thread block width
#define LAPLACE_W      56 

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4 

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts


//********************************Part 1  some constant var*************************************

//********************************Part 2   Cuda Image    ********************************************** //cudaImage.h
class CudaImage {
public:
	int width, height;
	int pitch;
	float *h_data;
	float *d_data;
	float *t_data;
	bool d_internalAlloc;
	bool h_internalAlloc;
public:
	CudaImage();
	~CudaImage();
	void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
	double Download();
	double Readback();
	double InitTexture();
	double CopyToTexture(CudaImage &dst, bool host);
};

extern int iDivUp(int a, int b);
extern int iDivDown(int a, int b);
extern int iAlignUp(int a, int b);
extern int iAlignDown(int a, int b);
extern void StartTimer(unsigned int *hTimer);
extern double StopTimer(unsigned int hTimer);



//*********************************** Part 2   Cuda Image *********************************************


//*********************************** Part 3   Cuda Sift **********************************************  // cudasift.h
typedef struct {
	float xpos;
	float ypos;
	float scale;
	float sharpness;
	float edgeness;
	float orientation;
	float score;
	float ambiguity;
	int match;
	float match_xpos;
	float match_ypos;
	float match_error;
	float subsampling;
	float empty[3];
	float data[128];
} SiftPoint;

typedef struct {
	int numPts;         // Number of available Sift points
	int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
	SiftPoint *m_data;    // Managed data
#else
	SiftPoint *h_data;  // Host (CPU) data
	SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;


//*************************************Part 4 cudasifth.h ***********************************************  //cudasifth.h

extern void ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
extern void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp);
extern double ScaleDown(CudaImage &res, CudaImage &src, float variance);
extern double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts);
extern double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling);
extern double LaplaceMulti(cudaTextureObject_t texObj, CudaImage *results, float baseBlur, float diffScale, float initBlur);
extern double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling);

extern void InitCuda(int devNum = 0);
extern void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
extern void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
extern void FreeSiftData(SiftData &data);
extern void PrintSiftData(SiftData &data);
extern double MatchSiftData(SiftData &data1, SiftData &data2);
extern double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);


//***********************************

//*********************************** Part 5 some IO

extern  void tk_write_Sift_Mat(SiftData* m_sift, const std::string& dumpname);









#endif