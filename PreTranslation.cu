#include <cutil_inline.h>

#define N_BINS				16
#define DIV_VAL_HIST		16	// GRAY_LEVELS / 16 = 256 / 16

//#define N_BINS			8
//#define DIV_VAL_HIST		32

#define N_THREADS_TRANS		16


unsigned int *d_img0, *d_img1;

unsigned int *d_horHists0, *d_horHists1;
unsigned int *d_verHists0, *d_verHists1;

int *h_trans, *d_trans;

float *h_diffHorHists, *d_diffHorHists;
float *h_diffVerHists, *d_diffVerHists;

int dx, dy;
float dxDiff, dyDiff;


// ========================================= PRIVATE UTILS =========================================


void checkErrorTrans(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);    
	}
}


// ========================================= KERNELS =========================================


__global__ void kernel_initHorHists(unsigned int *horHists0, unsigned int *horHists1) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	extern __shared__ int smem_initHorHists[];
	smem_initHorHists[threadIdx.x] = 0;
	__syncthreads();

	horHists0[globalIdx] = smem_initHorHists[threadIdx.x];
	horHists1[globalIdx] = smem_initHorHists[threadIdx.x];
}


__global__ void kernel_initVerHists(unsigned int *verHists0, unsigned int *verHists1) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	extern __shared__ int smem_initVerHists[];
	smem_initVerHists[threadIdx.x] = 0;
	__syncthreads();

	verHists0[globalIdx] = smem_initVerHists[threadIdx.x];
	verHists1[globalIdx] = smem_initVerHists[threadIdx.x];
}


__global__ void kernel_preCalcHorAndVerHists(unsigned int *horHists0, unsigned int *horHists1, unsigned int *verHists0, unsigned int *verHists1, 
											 unsigned int *img0, unsigned int *img1, int imH, int imW) {
	int globalIdx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x; // globalIdx = image position
	int binIdx0 = img0[globalIdx] / DIV_VAL_HIST;
	int binIdx1 = img1[globalIdx] / DIV_VAL_HIST;

	int y = globalIdx / imW;	int horHistId = __mul24(y, N_BINS);		atomicAdd(&horHists0[horHistId + binIdx0], 1);		atomicAdd(&horHists1[horHistId + binIdx1], 1);
	int x = globalIdx % imW;	int verHistId = __mul24(x, N_BINS);		atomicAdd(&verHists0[verHistId + binIdx0], 1);		atomicAdd(&verHists1[verHistId + binIdx1], 1);
}


__global__ void kernel_calcDiffHorHists(float *diffHorHists, unsigned int *horHists0, unsigned int *horHists1, int imH, int imW, 
										int *trans, int nTrans) { 
	int globalIdx	= __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int iThTrans	= globalIdx / imH;
	int iThRow		= globalIdx % imH;
	int dy			= trans[iThTrans];

	int nHorHists	= imH - abs(dy);

	if (iThRow < nHorHists) {
		int diff = 0;
		int absDy = __mul24(abs(dy), N_BINS);

		int horHistId0 = -1;
		int horHistId1 = -1;
		if (dy < 0) {
			horHistId0 = __mul24(iThRow, N_BINS);
			horHistId1 = horHistId0 + absDy;
		} else {
			horHistId1 = __mul24(iThRow, N_BINS);
			horHistId0 = horHistId1 + absDy;
		}

		#pragma unroll
		for(int i = 0; i < N_BINS; i++) {
			int d = horHists0[horHistId0 + i] - horHists1[horHistId1 + i];
			diff += abs(d);
		}
		
		extern __shared__ float smem_diffHorHists[];
		smem_diffHorHists[threadIdx.x] = 1 - diff*0.5/imW;
		__syncthreads();

		diffHorHists[globalIdx] = smem_diffHorHists[threadIdx.x];
	}
}


__global__ void kernel_calcDiffVerHists(float *diffVerHists, unsigned int *verHists0, unsigned int *verHists1, int imH, int imW,
										int *trans, int nTrans) {
	int globalIdx	= __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int iThTrans	= globalIdx / imW;
	int iThColumn	= globalIdx % imW;
	int dx			= trans[iThTrans];

	int nVerHists	= imW - abs(dx);

	if (iThColumn < nVerHists) {
		int diff = 0;
		int absDx = __mul24(abs(dx), N_BINS);

		int verHistId0 = -1; 
		int verHistId1 = -1; 
		if (dx < 0) {
			verHistId0 = __mul24(iThColumn, N_BINS);
			verHistId1 = verHistId0 + absDx;
		} else {
			verHistId1 = __mul24(iThColumn, N_BINS);
			verHistId0 = verHistId1 + absDx;
		}

		#pragma unroll
		for(int i = 0; i < N_BINS; i++) {
			int d = verHists0[verHistId0 + i] - verHists1[verHistId1 + i];
			diff += abs(d);
		}
		
		extern __shared__ float smem_diffVerHists[];
		smem_diffVerHists[threadIdx.x] = 1 - diff*0.5/imH;
		__syncthreads();

		diffVerHists[globalIdx] = smem_diffVerHists[threadIdx.x];
	}
}


// ==========================================================================================================================


void initTrans(int *img0, int *img1, int imH, int imW,
			   int transFrom, int transTo, int transRes, int nTrans) {

	int fourBytes		= sizeof(int);


	// ----- init arrays representing for 2 input images -----
	size_t memSizeOfImage = imH * imW * fourBytes;
	cudaMalloc((void **) &d_img0, memSizeOfImage);			cudaMemcpy(d_img0, img0, memSizeOfImage, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_img1, memSizeOfImage);			cudaMemcpy(d_img1, img1, memSizeOfImage, cudaMemcpyHostToDevice);	


	// ----- init horizontal and vertical histograms -----
	size_t memSizeOfHorHists = N_BINS * imH * fourBytes;	cudaMalloc((void **) &d_horHists0, memSizeOfHorHists);				cudaMalloc((void **) &d_horHists1, memSizeOfHorHists);
	size_t memSizeOfVerHists = N_BINS * imW * fourBytes;	cudaMalloc((void **) &d_verHists0, memSizeOfVerHists);				cudaMalloc((void **) &d_verHists1, memSizeOfVerHists);	


	// ----- init an array of trans -----
	size_t memSizeOfNTrans = nTrans * fourBytes;
	h_trans = (int *)malloc(memSizeOfNTrans);

	int count = 0;
	for(int i = transFrom; i <= transTo; i += transRes) 
		h_trans[count++] = i;

	cudaMalloc((void **) &d_trans, memSizeOfNTrans);
	cudaMemcpy(d_trans, h_trans, memSizeOfNTrans, cudaMemcpyHostToDevice);


	// ----- init diffHor and diffVer histograms -----
	size_t memSizeOfDiffHorHists = nTrans * imH * fourBytes;		cudaMalloc((void **) &d_diffHorHists, memSizeOfDiffHorHists);		h_diffHorHists = (float *)malloc(memSizeOfDiffHorHists);
	size_t memSizeOfDiffVerHists = nTrans * imW * fourBytes;		cudaMalloc((void **) &d_diffVerHists, memSizeOfDiffVerHists);		h_diffVerHists = (float *)malloc(memSizeOfDiffVerHists);
}


void preCalcHorAndVerHists(int imH, int imW) {
	int nThreads, nBlocks;
	unsigned int smemSize;


	// ------ init HOR and VER histograms ------
	nThreads = N_BINS;
	nBlocks = imH;
	smemSize = nThreads * sizeof(int);

	// init HOR hists	
	kernel_initHorHists<<<nBlocks, nThreads, smemSize>>>(d_horHists0, d_horHists1);		checkErrorTrans("kernel_initHorHists");
	cudaThreadSynchronize();															checkErrorTrans("kernel_initHorHists cudaThreadSynchronize");

	// init VER hists
	nBlocks = imW;
	kernel_initVerHists<<<nBlocks, nThreads, smemSize>>>(d_verHists0, d_verHists1);		checkErrorTrans("kernel_initVerHists");
	cudaThreadSynchronize();															checkErrorTrans("kernel_initVerHists cudaThreadSynchronize");


	// ------ pre-calculate HOR and VER histograms ------
	nThreads = N_THREADS_TRANS;
	nBlocks = (imH * imW) / nThreads; // dung cho anh co kich thuoc chia het cho 16
	kernel_preCalcHorAndVerHists<<<nBlocks, nThreads>>>(d_horHists0, d_horHists1, d_verHists0, d_verHists1, d_img0, d_img1, imH, imW);	checkErrorTrans("kernel_preCalcHorAndVerHists");
	cudaThreadSynchronize();																											checkErrorTrans("kernel_preCalcHorAndVerHists cudaThreadSynchronize");
}


void calcDiffHorHists(int imH, int imW, int nTrans) { 
	int nThreads = N_THREADS_TRANS;
	int nBlocks = (imH / nThreads) * nTrans;
	unsigned int smemSize = nThreads * sizeof(float); // smem contains 64 bytes

	kernel_calcDiffHorHists<<<nBlocks, nThreads, smemSize>>>(d_diffHorHists, d_horHists0, d_horHists1, imH, imW, d_trans, nTrans);	checkErrorTrans("kernel_calcDiffHorHists");
	cudaThreadSynchronize();																										checkErrorTrans("kernel_calcDiffHorHists cudaThreadSynchronize");

	cudaMemcpy(h_diffHorHists, d_diffHorHists, nTrans * imH * sizeof(float), cudaMemcpyDeviceToHost);								checkErrorTrans("copy from DEV to HOST");
}


void calcDiffVerHists(int imH, int imW, int nTrans) {
	int nThreads = N_THREADS_TRANS;
	int nBlocks = (imW / nThreads) * nTrans;
	unsigned int smemSize = nThreads * sizeof(float); // smem contains 64 bytes

	kernel_calcDiffVerHists<<<nBlocks, nThreads, smemSize>>>(d_diffVerHists, d_verHists0, d_verHists1, imH, imW, d_trans, nTrans);	checkErrorTrans("kernel_calcDiffVerHists");
	cudaThreadSynchronize();																										checkErrorTrans("kernel_calcDiffVerHists cudaThreadSynchronize");

	cudaMemcpy(h_diffVerHists, d_diffVerHists, nTrans * imW * sizeof(float), cudaMemcpyDeviceToHost);								checkErrorTrans("copy from DEV to HOST");
}


float calcGlobalDiffHorHists(int imH, int iThTrans) {
	int nHorHists = imH - abs(iThTrans);

	float diff = 0;
	for(int i = 0; i < nHorHists; i++) 	diff += h_diffHorHists[iThTrans * imH + i];
	return diff / nHorHists;
}


float calcGlobalDiffVerHists(int imW, int iThTrans) {
	int nVerHists = imW - abs(iThTrans);

	float diff = 0;
	for(int i = 0; i < nVerHists; i++) 	diff += h_diffVerHists[iThTrans * imW + i];
	return diff / nVerHists;
}


// --------------------------------------------------------------------


/*void estimateTrans(int *img0, int *img1, int imH, int imW,
				   int transFrom, int transTo, int transRes) {

	// initialization
	int nTrans = (transTo - transFrom) / transRes + 1;
	initTrans(img0, img1, imH, imW, transFrom, transTo, transRes, nTrans);
	preCalcHorAndVerHists(imH, imW);


	// calculate dx & dy
	calcDiffVerHists(imH, imW, nTrans);
	calcDiffHorHists(imH, imW, nTrans);

	dxDiff = -1.0;
	dyDiff = -1.0;
	for(int iThTrans = 0; iThTrans < nTrans; iThTrans++) {		
		float diffVer = calcGlobalDiffVerHists(imW, iThTrans);
		float diffHor = calcGlobalDiffHorHists(imH, iThTrans);
		if (dxDiff < diffVer) {		dxDiff = diffVer;		dx = transFrom + iThTrans*transRes;		}
		if (dyDiff < diffHor) {		dyDiff = diffHor;		dy = transFrom + iThTrans*transRes;		}
	}
}*/


void releaseTrans() {
	cudaFree(d_img0);		cudaFree(d_img1);

	cudaFree(d_horHists0);	cudaFree(d_horHists1);
	cudaFree(d_verHists0);	cudaFree(d_verHists1);

	free(h_trans);			cudaFree(d_trans);

	free(h_diffHorHists);	cudaFree(d_diffHorHists);
	free(h_diffVerHists);	cudaFree(d_diffVerHists);
}


// --------------------------------------------------------------------


extern "C" int getDx() {	return -dx;	}
extern "C" int getDy() {	return -dy;	}


extern "C" float runTransEst(int *img0, int *img1, int imH, int imW, 
							 int transFrom, int transTo, int transRes) {

	// ----------------- initialization -----------------
	int nTrans = (transTo - transFrom) / transRes + 1;
	initTrans(img0, img1, imH, imW, transFrom, transTo, transRes, nTrans);
	preCalcHorAndVerHists(imH, imW);
	// --------------------------------------------------


		// computation time measure
		cudaEvent_t timeStart, timeStop;

		float elapsedTime;
		cudaEventCreate(&timeStart);
		cudaEventCreate(&timeStop);
		cudaEventRecord(timeStart, 0);


	// ----------------- calculate dx & dy -----------------
	calcDiffVerHists(imH, imW, nTrans);
	calcDiffHorHists(imH, imW, nTrans);

	dxDiff = -1.0;
	dyDiff = -1.0;
	for(int iThTrans = 0; iThTrans < nTrans; iThTrans++) {		
		float diffVer = calcGlobalDiffVerHists(imW, iThTrans);
		float diffHor = calcGlobalDiffHorHists(imH, iThTrans);
		if (dxDiff < diffVer) {		dxDiff = diffVer;		dx = transFrom + iThTrans*transRes;		}
		if (dyDiff < diffHor) {		dyDiff = diffHor;		dy = transFrom + iThTrans*transRes;		}
	}
	// -----------------------------------------------------


		cudaEventRecord(timeStop, 0);
		cudaEventSynchronize(timeStop);
		cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);
		cudaEventDestroy(timeStart);
		cudaEventDestroy(timeStop);

		// print results to file
		FILE *fp;
		fp = fopen("result.txt", "a");
		//fprintf(fp, "dx=%d, dxDiff=%f,   dy=%d, dyDiff=%f,   t=%f \n", getDx(), dxDiff, getDy(), dyDiff, elapsedTime);
		fprintf(fp, "dx=%d, dy=%d, t=%1.3f   ", getDx(), getDy(), elapsedTime);
		fclose(fp);


	releaseTrans();

	return elapsedTime;
}