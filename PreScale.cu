#include <cutil_inline.h>

/*#define N_BINS			16
#define DIV_VAL_HIST		16	// GRAY_LEVELS / 16 = 256 / 16

#define N_THREADS_SCALE		16
#define VER_HIST_SMEM_SIZE	256 // N_BINS x N_THREADS_SCALE = 16 x 16
*/

#define N_BINS				8
#define DIV_VAL_HIST		32	

#define N_THREADS_SCALE		16
#define VER_HIST_SMEM_SIZE	128


unsigned int *d_im0, *d_im1;
float *h_scales, *d_scales;
float *h_diffScaleVerHists, *d_diffScaleVerHists;

float sy;
float syDiff;


// ========================================= PRIVATE UTILS =========================================


void checkErrorScale(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);    
	}
}


// ========================================= KERNELS =========================================


__global__ void kernel_calcDiffScaleVerHists(float *diffScaleVerHists, unsigned int *im0, unsigned int *im1, int imH, int imW, 
											 float *scales, int nScales) { // scale < 1: zoom out, scale > 1: zoom in

	int globalIdx	= __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int iThScale	= globalIdx / imW;
	int iThColumn	= globalIdx % imW;
	float scale		= scales[iThScale];


	// ---------- define shared memories ----------
	__shared__ int smem_verHist0[VER_HIST_SMEM_SIZE]; // with size = N_BINS * VER_HIST_N_THREADS_SCALE = 16 bins * 16 threads = 256 (one thread processes one column)
	__shared__ int smem_verHist1[VER_HIST_SMEM_SIZE];

	__shared__ float smem_diffVerHist[N_THREADS_SCALE];


	// init shared memories to 0
	#pragma unroll
	for(int i = 0; i < N_BINS; i++) {
		int smemPos = __mul24(i, N_THREADS_SCALE) + threadIdx.x;
		smem_verHist0[smemPos] = 0;
		smem_verHist1[smemPos] = 0;
	}
	__syncthreads();



	// ---------- calculate vertical histograms of 2 images and store to shared memories ----------
	int interpolatedImH = -1;
	int startPos = -1;

	// determine zoom out or zoom in
	float scaleUp = 1 - (scale - 1);

	if(scale < 1) { // zoom out
		interpolatedImH = int(imH * scale);		// for im0			// interpolatedImgH < imH
		startPos = int(imH * (1 - scale)/2);	// for im1
	} else { // zoom in
		startPos = int(imH * (1 - scaleUp)/2);	// for im0
		interpolatedImH = int(imH * scaleUp); // for img1	
	}

	// calc hist.
	int tmp = __mul24(startPos, imW) + iThColumn;

	#pragma unroll
	for(int i = 0; i < interpolatedImH; i++) {
		int interpolatedI = -1;

		if(scale < 1) { // zoom out
			interpolatedI = int(i/scale); // i * (1 / scale)		// interpolatedI > i
		} else { // zoom in
			interpolatedI = int(i/scaleUp);
		}

		int imPos0 = __mul24(interpolatedI, imW) + iThColumn;		
		int imPos1 = __mul24(i, imW) + tmp;
		//int imgPos1 = (i + startPos) * imgW + idx; // = i*imgW + startPos*imgW + idx = i*imgW + tmp;

		int binIdx0 = im0[imPos0] / DIV_VAL_HIST;		int smemPos0 = __mul24(binIdx0, N_THREADS_SCALE) + threadIdx.x;		smem_verHist0[smemPos0]++;
		int binIdx1 = im1[imPos1] / DIV_VAL_HIST;		int smemPos1 = __mul24(binIdx1, N_THREADS_SCALE) + threadIdx.x;		smem_verHist1[smemPos1]++;
	}
	__syncthreads();



	// ---------- calculate the difference btw vertical histograms and store to global memory ----------
	int diff = 0;

	#pragma unroll
	for(int i = 0; i < N_BINS; i++) {
		int smemPos = __mul24(i, N_THREADS_SCALE) + threadIdx.x;
		diff += abs(smem_verHist0[smemPos] - smem_verHist1[smemPos]);
	}
	smem_diffVerHist[threadIdx.x] = 1 - diff*0.5/imH;
	__syncthreads();

	diffScaleVerHists[globalIdx] = smem_diffVerHist[threadIdx.x];
}


// ==========================================================================================================================


void initScale(int *img0, int *img1, int imH, int imW,
			   double scaleFrom, double scaleTo, double scaleRes, int nScales) {
    
	int fourBytes = sizeof(int);
	

	// ----- init arrays representing for 2 input images -----
	size_t memSizeOfImage = imH * imW * fourBytes;
	cudaMalloc((void **) &d_im0, memSizeOfImage);					cudaMemcpy(d_im0, img0, memSizeOfImage, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_im1, memSizeOfImage);					cudaMemcpy(d_im1, img1, memSizeOfImage, cudaMemcpyHostToDevice);	


	// ----- init an array of scales -----
	size_t memSizeOfNScales = nScales * fourBytes;
	h_scales = (float *)malloc(memSizeOfNScales);

	int count = 0;
	for(double i = scaleFrom; i <= scaleTo; i += scaleRes) // i < 1: zoom out, i > 1: zoom in
		h_scales[count++] = float(i);

	cudaMalloc((void **) &d_scales, memSizeOfNScales);
	cudaMemcpy(d_scales, h_scales, memSizeOfNScales, cudaMemcpyHostToDevice);


	// ----- init array for the different btw 2 vertical histograms -----
	size_t memSizeOfDiffScaleVerHists = nScales * imW * fourBytes;
	cudaMalloc((void **) &d_diffScaleVerHists, memSizeOfDiffScaleVerHists);
	h_diffScaleVerHists = (float *)malloc(memSizeOfDiffScaleVerHists);
}


void calcDiffScaleVerHists(int imH, int imW, int nScales) {
	int nThreads = N_THREADS_SCALE;
	int nBlocks = (imW / nThreads) * nScales; // imageWidth chia het cho 16

	kernel_calcDiffScaleVerHists<<<nBlocks, nThreads>>>(d_diffScaleVerHists, d_im0, d_im1, imH, imW, d_scales, nScales);	checkErrorScale("kernel_calcDiffScaleVerHists");
	cudaThreadSynchronize();																								checkErrorScale("kernel_calcDiffScaleVerHists cudaThreadSynchronize");

	size_t memSizeOfDiffScaleVerHists = nScales * imW * sizeof(float);
	cudaMemcpy(h_diffScaleVerHists, d_diffScaleVerHists, memSizeOfDiffScaleVerHists, cudaMemcpyDeviceToHost);				checkErrorScale("copy from DEV to HOST");
}


float calcGlobalDiffScaleVerHist(int imW, int iThScale) {
	float diff = 0;
	for(int i = 0; i < imW; i++) 
		diff += h_diffScaleVerHists[iThScale * imW + i];
	return diff / imW;
}


// --------------------------------------------------------------------


/*void estimateScale(int *img0, int *img1, int imH, int imW,
				   double scaleFrom, double scaleTo, double scaleRes) {

	// initialization
	int nScales = int((scaleTo - scaleFrom) / scaleRes + 1);
	initScale(img0, img1, imH, imW, scaleFrom, scaleTo, scaleRes, nScales); // co phan nay la ton > 30 % thoi gian

	// calculate sy
	calcDiffScaleVerHists(imH, imW, nScales);

	syDiff = -1.0;
	for(int iThScale = 0; iThScale < nScales; iThScale++) { // i < 1: zoom out, i > 1: zoom in
		float diff = calcGlobalDiffScaleVerHist(imW, iThScale);
		if (syDiff < diff) {
			syDiff = diff;
			sy = float(scaleFrom + iThScale*scaleRes);
		}
	}
}*/


void releaseScale() {
	cudaFree(d_im0);			cudaFree(d_im1);
	free(h_scales);				cudaFree(d_scales);
	free(h_diffScaleVerHists);	cudaFree(d_diffScaleVerHists);
}


// --------------------------------------------------------------------


extern "C" float getSy() {	return sy;	}


extern "C" float runScaleEst(int *img0, int *img1, int imH, int imW,
							 double scaleFrom, double scaleTo, double scaleRes) {

	// ---------------------- initialization ----------------------
	int nScales = int((scaleTo - scaleFrom) / scaleRes + 1);
	initScale(img0, img1, imH, imW, scaleFrom, scaleTo, scaleRes, nScales); // co phan nay la ton > 30 % thoi gian
	// ------------------------------------------------------------


		// computation time measure
		cudaEvent_t timeStart, timeStop;

		float elapsedTime;
		cudaEventCreate(&timeStart);
		cudaEventCreate(&timeStop);
		cudaEventRecord(timeStart, 0);
	

	// ---------------------- calculate sy ----------------------
	calcDiffScaleVerHists(imH, imW, nScales);

	syDiff = -1.0;
	for(int iThScale = 0; iThScale < nScales; iThScale++) { // i < 1: zoom out, i > 1: zoom in
		float diff = calcGlobalDiffScaleVerHist(imW, iThScale);
		if (syDiff < diff) {
			syDiff = diff;
			sy = float(scaleFrom + iThScale*scaleRes);
		}
	}
	// -----------------------------------------------------------

	
		cudaEventRecord(timeStop, 0);
		cudaEventSynchronize(timeStop);
		cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);
		cudaEventDestroy(timeStart);
		cudaEventDestroy(timeStop);

		// print results to file
		FILE *fp;
		fp = fopen("result.txt", "a");
		//fprintf(fp, "sy=%f, syDiff=%f,   t=%f \n", getSy(), syDiff, elapsedTime);
		fprintf(fp, "sy=%1.3f, t=%1.3f   ", getSy(), elapsedTime);
		fclose(fp);


	releaseScale();

	return elapsedTime;
}