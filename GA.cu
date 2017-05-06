#include "cudpp/cudpp.h"
#include <cutil_inline.h>
#include <vector_types.h>

#define MUL_FACTOR_10K		(10000)
#define MUL_FACTOR_10001	(10001)


unsigned int *d_image0, *d_image1;

uint4 *d_chromosomes_int;
float4 *d_chromosomes;
float4 *h_chromosomes;	// for debug

int *d_diffIntensities, *d_diffIntensities_count;
int *d_sumOfDiffIntensities, *d_sumOfDiffIntensities_count;
int *h_sumOfDiffIntensities; // for debug

float *d_errors; // fitness values
float *h_errors; // for debug

uint4 *d_randNumForCrossover_int;
float4 *d_randNumForCrossover;

int bestErrorIdx;


// ========================================= PRIVATE UTILS =========================================


void checkErrorGA(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);    
	}
}


// ========================================= KERNELS =========================================


// ------------- for createPopulation -------------


__global__ void kernel_convertIntPopToFloatPop(float4 *chromosomes, uint4 *chromosomes_int,
											   int sRangeSize, int dxRangeSize, int dyRangeSize, 
											   float sRangeFrom, int dxRangeFrom, int dyRangeFrom) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float sx = float(chromosomes_int[globalIdx].x % sRangeSize) / MUL_FACTOR_10K + sRangeFrom;
	float sy = float(chromosomes_int[globalIdx].y % sRangeSize) / MUL_FACTOR_10K + sRangeFrom;
	float dx = float(chromosomes_int[globalIdx].z % dxRangeSize) / MUL_FACTOR_10K + dxRangeFrom;
	float dy = float(chromosomes_int[globalIdx].w % dyRangeSize) / MUL_FACTOR_10K + dyRangeFrom;

	extern __shared__ float4 smem_floatPop[];
	smem_floatPop[threadIdx.x] = make_float4(sx, sy, dx, dy);
	__syncthreads();

	chromosomes[globalIdx] = smem_floatPop[threadIdx.x];
}


// ------------- for calculating fitness values -------------


__global__ void kernel_initDiffIntensities(int *diffIntensities, int *diffIntensities_count) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	extern __shared__ int smem_initDiffIntensities[];
	smem_initDiffIntensities[threadIdx.x] = 0;
	__syncthreads();

	diffIntensities[globalIdx] = smem_initDiffIntensities[threadIdx.x];
	diffIntensities_count[globalIdx] = smem_initDiffIntensities[threadIdx.x];
}


__global__ void kernel_calcDiffIntensities(int *diffIntensities, int *diffIntensities_count,
										   unsigned int *image0, unsigned int *image1, int imSize, int imH, int imW, int imHdiv2, int imWdiv2,
										   float4 *chromosomes, int popSize) {

	int globalIdx		= __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int iThChromosome	= globalIdx / imSize;
	int iThPixel		= globalIdx % imSize;	

	// sx = a1, sy = a4, dx = b1, dy = b2
	float a1 = chromosomes[iThChromosome].x;
	float a4 = chromosomes[iThChromosome].y;
	float b1 = chromosomes[iThChromosome].z;
	float b2 = chromosomes[iThChromosome].w;

	// truoc tien phai tinh nghich dao cho ma tran A (a1, a2 = a3 = 0, a4) va B (b1, b2)
	double a1_ = 1/a1;		double a4_ = 1/a4;
	double b1_ = a1_*b1;	double b2_ = a4_*b2; 

	// ----- main run -----
	int x = iThPixel % imW;		int x_imWdiv2 = x - imWdiv2;		float newX = a1_ * x_imWdiv2 - b1_;		newX += imWdiv2;			int newX_ = int(newX);
	int y = iThPixel / imW;		int y_imHdiv2 = y - imHdiv2;		float newY = a4_ * y_imHdiv2 - b2_;		newY += imHdiv2;			int newY_ = int(newY);

	__shared__ int smem_diffIntensities[512];
	__shared__ int smem_diffIntensities_count[512];

	if(0 <= newX_ && newX_ < imW - 1 && 0 <= newY_ && newY_ < imH - 1) { // minimum neighborhood
		int pos = y * imW + x;
		int des0 = image0[ newY_	  * imW	+  newX_     ] 	- image1[pos];		if(des0 < 0) des0 = -des0;
		int des1 = image0[ newY_	  * imW	+ (newX_ + 1)]	- image1[pos];		if(des1 < 0) des1 = -des1;		if(des0 > des1) des0 = des1;
		    des1 = image0[(newY_ + 1) * imW	+  newX_     ]	- image1[pos];		if(des1 < 0) des1 = -des1;		if(des0 > des1) des0 = des1;
			des1 = image0[(newY_ + 1) * imW + (newX_ + 1)]	- image1[pos];		if(des1 < 0) des1 = -des1;		if(des0 > des1) des0 = des1;
		
		smem_diffIntensities[threadIdx.x] = des0;
		smem_diffIntensities_count[threadIdx.x] = 1; // count++ (co tinh pixel nay)
		__syncthreads();
		
		diffIntensities[globalIdx] = smem_diffIntensities[threadIdx.x];
		diffIntensities_count[globalIdx] = smem_diffIntensities_count[threadIdx.x];
	}
}


__global__ void kernel_calcErrors(float *errors, int *sumOfDiffIntensities, int *sumOfDiffIntensities_count, int imSize) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int sumIdx = (globalIdx + 1) * imSize - 1;

	extern __shared__ float smem_errors[];
	if(globalIdx == 0) 		
		smem_errors[threadIdx.x] = sumOfDiffIntensities[sumIdx] * 1.0 / sumOfDiffIntensities_count[sumIdx];
	else 
		smem_errors[threadIdx.x] = (sumOfDiffIntensities[sumIdx] - sumOfDiffIntensities[sumIdx - imSize]) * 1.0 / (sumOfDiffIntensities_count[sumIdx] - sumOfDiffIntensities_count[sumIdx - imSize]);
	__syncthreads();

	errors[globalIdx] = smem_errors[threadIdx.x];
}


// ------------- for tournament selection -------------


__global__ void kernel_selection(float4 *chromosomes, float *errors) {
	int globalIdx	= __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float iThError	= errors[globalIdx];
	int randN		= int(iThError * 1000) % blockDim.x;

	extern __shared__ float4 smem_selection[];

	if(iThError > errors[randN]) {
		smem_selection[threadIdx.x] = chromosomes[randN];
		smem_selection[randN]		= chromosomes[randN];
	} else {
		smem_selection[threadIdx.x] = chromosomes[threadIdx.x];
		smem_selection[randN]		= chromosomes[threadIdx.x];
	}
	__syncthreads();

	chromosomes[globalIdx] = smem_selection[threadIdx.x];
}


// ------------- for uniform crossover -------------


__global__ void kernel_convertRandIntToRandFloat(float4 *randNumForCrossover, uint4 *randNumForCrossover_int) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	extern __shared__ float4 smem_randFloat[];
	smem_randFloat[threadIdx.x].x = float(randNumForCrossover_int[globalIdx].x % MUL_FACTOR_10001) / MUL_FACTOR_10K;
	smem_randFloat[threadIdx.x].y = float(randNumForCrossover_int[globalIdx].y % MUL_FACTOR_10001) / MUL_FACTOR_10K;
	smem_randFloat[threadIdx.x].z = float(randNumForCrossover_int[globalIdx].z % MUL_FACTOR_10001) / MUL_FACTOR_10K;
	smem_randFloat[threadIdx.x].w = float(randNumForCrossover_int[globalIdx].w % MUL_FACTOR_10001) / MUL_FACTOR_10K;
	__syncthreads();

	randNumForCrossover[globalIdx] = smem_randFloat[threadIdx.x];
}


__global__ void kernel_crossover(float4 *chromosomes, float4 *randNumForCrossover, float pc, int halfPopSize) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int halfPopSize_globalIdx = globalIdx + halfPopSize;

	extern __shared__ float4 smem_forCrossover[];

	// sx
	float randForSx = randNumForCrossover[globalIdx].x;
	if(randForSx < pc) {	smem_forCrossover[threadIdx.x].x = chromosomes[halfPopSize_globalIdx].x;		smem_forCrossover[halfPopSize_globalIdx].x = chromosomes[threadIdx.x].x;
	} else {				smem_forCrossover[threadIdx.x].x = chromosomes[threadIdx.x].x;					smem_forCrossover[halfPopSize_globalIdx].x = chromosomes[halfPopSize_globalIdx].x;		}

	// sy
	float randForSy = randNumForCrossover[globalIdx].y;
	if(randForSy < pc) {	smem_forCrossover[threadIdx.x].y = chromosomes[halfPopSize_globalIdx].y;		smem_forCrossover[halfPopSize_globalIdx].y = chromosomes[threadIdx.x].y;
	} else {				smem_forCrossover[threadIdx.x].y = chromosomes[threadIdx.x].y;					smem_forCrossover[halfPopSize_globalIdx].y = chromosomes[halfPopSize_globalIdx].y;		}

	// dx
	float randForDx = randNumForCrossover[globalIdx].z;
	if(randForDx < pc) {	smem_forCrossover[threadIdx.x].z = chromosomes[halfPopSize_globalIdx].z;		smem_forCrossover[halfPopSize_globalIdx].z = chromosomes[threadIdx.x].z;
	} else {				smem_forCrossover[threadIdx.x].z = chromosomes[threadIdx.x].z;					smem_forCrossover[halfPopSize_globalIdx].z = chromosomes[halfPopSize_globalIdx].z;		}

	// dy
	float randForDy = randNumForCrossover[globalIdx].w;
	if(randForDy < pc) {	smem_forCrossover[threadIdx.x].w = chromosomes[halfPopSize_globalIdx].w;		smem_forCrossover[halfPopSize_globalIdx].w = chromosomes[threadIdx.x].w;
	} else {				smem_forCrossover[threadIdx.x].w = chromosomes[threadIdx.x].w;					smem_forCrossover[halfPopSize_globalIdx].w = chromosomes[halfPopSize_globalIdx].w;		}

	__syncthreads();

	chromosomes[globalIdx] = smem_forCrossover[threadIdx.x];
}


// ==========================================================================================================================


void initGA(int *img0, int *img1, int imH, int imW, int popSize) {
	int fourBytes		= sizeof(int);
	int sixteenBytes	= sizeof(int4);
	int imSize			= imH * imW;


	// ----- init arrays representing for 2 input images -----
	size_t memSizeOfImage = imSize * fourBytes;
	cudaMalloc((void **) &d_image0, memSizeOfImage);					cudaMemcpy(d_image0, img0, memSizeOfImage, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_image1, memSizeOfImage);					cudaMemcpy(d_image1, img1, memSizeOfImage, cudaMemcpyHostToDevice);


	// ----- init chromosome list (population) -----
	size_t memSizeOfChromosomes	= popSize * sixteenBytes;
	cudaMalloc((void **) &d_chromosomes_int, memSizeOfChromosomes); // for generating random integer numbers only
	cudaMalloc((void **) &d_chromosomes, memSizeOfChromosomes);
	h_chromosomes = (float4 *)malloc(memSizeOfChromosomes); // for debug


	// ----- init error list (for fitness calculation) -----
	size_t memSizeOfDiffIntensities = imSize * popSize * fourBytes;
	cudaMalloc((void **) &d_diffIntensities, memSizeOfDiffIntensities);			cudaMalloc((void **) &d_sumOfDiffIntensities, memSizeOfDiffIntensities);
	cudaMalloc((void **) &d_diffIntensities_count, memSizeOfDiffIntensities);	cudaMalloc((void **) &d_sumOfDiffIntensities_count, memSizeOfDiffIntensities);		
	h_sumOfDiffIntensities = (int *)malloc(memSizeOfDiffIntensities); // for debug

	size_t memSizeOfErrors = popSize * fourBytes;
	cudaMalloc((void **) &d_errors, memSizeOfErrors);	
	h_errors = (float *)malloc(memSizeOfErrors); // for debug


	// ----- int list of random numbers (for crossover) -----
	size_t memSizeOfHalfChromosomes	= (popSize/2) * sixteenBytes;
	cudaMalloc((void **) &d_randNumForCrossover_int, memSizeOfHalfChromosomes); // for generating random integer numbers only
	cudaMalloc((void **) &d_randNumForCrossover, memSizeOfHalfChromosomes); // for generating random integer numbers only
}


void createPopulation(int popSize, 
					  int sRangeSize, int dxRangeSize, int dyRangeSize, 
					  float sRangeFrom, int dxRangeFrom, int dyRangeFrom) {

	size_t lengthOfChromosomes = popSize * 4; // 4: because of float4


	// ----- generate random integer numbers using CUDPP -----	
	CUDPPConfiguration config;
	config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_RAND_MD5;
    config.options = CUDPP_OPTION_FORWARD;
    
    CUDPPHandle plan = 0;
    cudppPlan(&plan, config, lengthOfChromosomes, /* row */ 1, /* rowPitch */ 0);

	cudppRand(plan, d_chromosomes_int, lengthOfChromosomes);
	cudppDestroyPlan(plan);


	// ----- convert rand int numbers (d_chromosomes_int) to rand float numbers -----
	int nThreads = 16;
	int nBlocks = popSize / nThreads;
	unsigned int smemSize = nThreads * sizeof(float4);
	
	kernel_convertIntPopToFloatPop<<<nBlocks, nThreads, smemSize>>>(d_chromosomes, d_chromosomes_int, 
																	sRangeSize, dxRangeSize, dyRangeSize, 
																	sRangeFrom, dxRangeFrom, dyRangeFrom);		checkErrorGA("kernel_convertIntPopToFloatPop");
	cudaThreadSynchronize();																					checkErrorGA("kernel_convertIntPopToFloatPop cudaThreadSynchronize");	
}


void calcFitness(int imH, int imW, int popSize) {
	int imSize = imH * imW;
	int nPixels = imSize * popSize;

	int nThreads, nBlocks;
	unsigned int smemSize;


	// ----- init d_diffIntensities and d_diffIntensities_count to zeros -----
	nThreads = 512; // 640x480 va 320x240 deu chia het cho 512
	nBlocks = nPixels / nThreads;
	smemSize = nThreads * sizeof(int);

	kernel_initDiffIntensities<<<nBlocks, nThreads, smemSize>>>(d_diffIntensities, d_diffIntensities_count);	checkErrorGA("kernel_initDiffIntensities");
	cudaThreadSynchronize();																					checkErrorGA("kernel_initDiffIntensities cudaThreadSynchronize");


	// ----- calculate the difference of intensities of every pixel pair in the 2 images -----
	nThreads = 512;
	nBlocks = nPixels / nThreads;

	kernel_calcDiffIntensities<<<nBlocks, nThreads>>>(d_diffIntensities, d_diffIntensities_count,
													  d_image0, d_image1, imSize, imH, imW, imH / 2, imW / 2,
													  d_chromosomes, popSize);									checkErrorGA("kernel_calcDiffIntensities");
	cudaThreadSynchronize();																					checkErrorGA("kernel_calcDiffIntensities cudaThreadSynchronize");

	
	// ----- calculate sum of different intensities -----
	CUDPPConfiguration config;
	config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD;// | CUDPP_OPTION_INCLUSIVE;
	config.op = CUDPP_ADD;
    
    CUDPPHandle plan = 0;
    cudppPlan(&plan, config, nPixels, /* row */ 1, /* rowPitch */ 0);

	cudppScan(plan, d_sumOfDiffIntensities, d_diffIntensities, nPixels); // sum of different intensities
	cudppScan(plan, d_sumOfDiffIntensities_count, d_diffIntensities_count, nPixels); // sum of counts
	cudppDestroyPlan(plan);

	// for debug
	//cudaMemcpy(h_sumOfDiffIntensities, d_sumOfDiffIntensities, nPixels * sizeof(int), cudaMemcpyDeviceToHost);						checkErrorGA("copy from DEV to HOST");


	// ----- calculate errors based on d_sumOfDiffIntensities and d_sumOfDiffIntensities_count -----
	nThreads = 16;
	nBlocks = popSize / nThreads;
	smemSize = nThreads * sizeof(float);

	kernel_calcErrors<<<nBlocks, nThreads, smemSize>>>(d_errors, d_sumOfDiffIntensities, d_sumOfDiffIntensities_count, imSize);		checkErrorGA("kernel_calcErrors");
	cudaThreadSynchronize();																										checkErrorGA("kernel_calcErrors cudaThreadSynchronize");
}


void selection(int popSize) {
	int nThreads = popSize;
	int nBlocks = 1;
	unsigned int smemSize = nThreads * sizeof(float4);
	
	kernel_selection<<<nBlocks, nThreads, smemSize>>>(d_chromosomes, d_errors);		checkErrorGA("kernel_selection");
	cudaThreadSynchronize();														checkErrorGA("kernel_selection cudaThreadSynchronize");
}


void crossover(float pc, int popSize) {
	int halfPopSize = popSize / 2;
	size_t halfLengthOfChromosomes = halfPopSize * 4; // 4: because of float4


	// ----- generate random integer numbers using CUDPP -----	
	CUDPPConfiguration config;
	config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_RAND_MD5;
    config.options = CUDPP_OPTION_FORWARD;
    
    CUDPPHandle plan = 0;
    cudppPlan(&plan, config, halfLengthOfChromosomes, /* row */ 1, /* rowPitch */ 0);

	cudppRand(plan, d_randNumForCrossover_int, halfLengthOfChromosomes);
	cudppDestroyPlan(plan);


	// ----- convert rand int numbers (d_randNumForCrossover_int) to rand float numbers -----
	int nThreads = 16;
	int nBlocks = halfPopSize / nThreads; 
	unsigned int smemSize = nThreads * sizeof(float4);
	
	kernel_convertRandIntToRandFloat<<<nBlocks, nThreads, smemSize>>>(d_randNumForCrossover, d_randNumForCrossover_int);	checkErrorGA("kernel_convertRandIntToRandFloat");
	cudaThreadSynchronize();																								checkErrorGA("kernel_convertRandIntToRandFloat cudaThreadSynchronize");

	// for debug
	//cudaMemcpy(h_chromosomes, d_randNumForCrossover, halfPopSize * sizeof(float4), cudaMemcpyDeviceToHost);					checkErrorGA("copy from DEV to HOST");	


	// ----- crossover -----
	nThreads = halfPopSize;
	nBlocks = 1;
	smemSize = popSize * sizeof(float4);

	kernel_crossover<<<nBlocks, nThreads, smemSize>>>(d_chromosomes, d_randNumForCrossover, pc, halfPopSize);		checkErrorGA("kernel_crossover");
	cudaThreadSynchronize();																						checkErrorGA("kernel_crossover cudaThreadSynchronize");
}


void releaseGA() {
	cudaFree(d_image0);						cudaFree(d_image1);
	cudaFree(d_chromosomes_int);			cudaFree(d_chromosomes);				free(h_chromosomes);				
	cudaFree(d_diffIntensities);			cudaFree(d_sumOfDiffIntensities);		free(h_sumOfDiffIntensities);
	cudaFree(d_diffIntensities_count);		cudaFree(d_sumOfDiffIntensities_count);
	cudaFree(d_errors);																free(h_errors);
	cudaFree(d_randNumForCrossover_int);	cudaFree(d_randNumForCrossover);
}


// ==========================================================================================================================


extern "C" float getBestSx() {	return h_chromosomes[bestErrorIdx].x;	}
extern "C" float getBestSy() {	return h_chromosomes[bestErrorIdx].y;	}
extern "C" float getBestDx() {	return h_chromosomes[bestErrorIdx].z;	}
extern "C" float getBestDy() {	return h_chromosomes[bestErrorIdx].w;	}

extern "C" float getBestError() {	return h_errors[bestErrorIdx];		}


extern "C" float runGA(int *img0, int *img1, int imH, int imW,
					   /* GA parameters */		int popSize, double pc, int nGen,
					   /* translation range */	int dxRangeFrom, int dxRangeTo, int dyRangeFrom, int dyRangeTo,
					   /* scale range */		double sRangeFrom, double sRangeTo) {

    // ------------------------------ init -------------------------------
	int sRangeSize = int((sRangeTo - sRangeFrom) * MUL_FACTOR_10K + 1);
	int dxRangeSize = int((dxRangeTo - dxRangeFrom) * MUL_FACTOR_10K + 1);
	int dyRangeSize = int((dyRangeTo - dyRangeFrom) * MUL_FACTOR_10K + 1);

	initGA(img0, img1, imH, imW, popSize);
	// -------------------------------------------------------------------


		// computation time measure
		cudaEvent_t timeStart, timeStop;

		float elapsedTime;
		cudaEventCreate(&timeStart);
		cudaEventCreate(&timeStop);
		cudaEventRecord(timeStart, 0);


	// ------------------------------ main run----------------------------
	createPopulation(popSize, 
					 sRangeSize, dxRangeSize, dyRangeSize, 
					 float(sRangeFrom), dxRangeFrom, dyRangeFrom);
	
	for(int i = 0; i < nGen; i++) {
		calcFitness(imH, imW, popSize);
		selection(popSize);
		crossover(float(pc), popSize);
	}

	calcFitness(imH, imW, popSize);
	// -------------------------------------------------------------------


		cudaEventRecord(timeStop, 0);
		cudaEventSynchronize(timeStop);
		cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);
		cudaEventDestroy(timeStart);
		cudaEventDestroy(timeStop);


	// ------------------------------ debug	& print ------------------------------
	cudaMemcpy(h_chromosomes, d_chromosomes, popSize * sizeof(float4), cudaMemcpyDeviceToHost);					checkErrorGA("copy from DEV to HOST");
	cudaMemcpy(h_errors, d_errors, popSize * sizeof(float), cudaMemcpyDeviceToHost);							checkErrorGA("copy from DEV to HOST");
	
	// getBestErrorIdx();
	float bestError = 1000;
	for(int i = 0; i < popSize; i++) {
		if(bestError > h_errors[i]) {
			bestError = h_errors[i];
			bestErrorIdx = i;
		}
	}

	// print "chromosomes" to file
/*	FILE *fp;
	fp = fopen("chromosomes.txt", "a");
	for(int i = 0; i < popSize; i++) 		fprintf(fp, "%d  %f  %f  %f  %f \n", i, h_chromosomes[i].x, h_chromosomes[i].y, h_chromosomes[i].z, h_chromosomes[i].w);
	fclose(fp);

	// print "errors-fitness values" to file
	fp = fopen("errors.txt", "a");
	for(int i = 0; i < popSize; i++)		fprintf(fp, "%d  %f \n", i, h_errors[i]);
	fclose(fp);*/
	// ----------------------------------------------------------------------------


	releaseGA();

	return elapsedTime;
}