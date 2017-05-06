#include "cv.h"
#include "highgui.h"

#include "PreTranslation.h"
#include "PreScale.h"
#include "GA.h"


// --------------------------

#define BATCH_N_IMG					2
#define RESULT_FILENAME				"result.txt"
//#define BATCH_IMG_FILENAME			".\\testdata\\tmp\\%04d.bmp"
#define BATCH_IMG_FILENAME			"..\\testdata\\100929_MOV06059\\1\\gray\\%04d.bmp"


#define TRANS_FROM		-20
#define TRANS_TO		20
#define TRANS_RES		1

#define SCALE_FROM		0.99
#define SCALE_TO		1.01
#define SCALE_RES		0.01

#define GA_POP_SIZE		128		// chia het cho 16
#define GA_PC			0.5
#define GA_N_GEN		3

#define GA_TRANS_RANGE	1		// = \Delta t_x = \Delta t_y
#define GA_SCALE_RANGE	0.01	// = \Delta s_x = \Delta s_y


// --------------------------

IplImage *srcImg0, *srcImg1, *grayImg0, *grayImg1;
int imW, imH;

int *img0, *img1;

// --------------------------

void convertCharImgToIntImg() {
	int memSizeOfImg = imW * imH * sizeof(int);
	img0 = (int *)malloc(memSizeOfImg);
	img1 = (int *)malloc(memSizeOfImg);

	const uchar* grayImg0Data = (uchar *) grayImg0->imageData;
	const uchar* grayImg1Data = (uchar *) grayImg1->imageData;

	for(int i = 0; i < imH; i++) {
		for(int j = 0; j < imW; j++) {
			int pos = i*imW + j;
			img0[pos] = (int) grayImg0Data[pos];
			img1[pos] = (int) grayImg1Data[pos];
		}
	}
}


void release() {
	cvReleaseImage(&srcImg0);	cvReleaseImage(&grayImg0);
	cvReleaseImage(&srcImg1);	cvReleaseImage(&grayImg1);
	delete img0;
	delete img1;
}


// ===========================================================================================================================


double calculateAvgIntensityWrtAffine(IplImage* src1, IplImage* src2, double a1, double a4, double b1, double b2) {

	// truoc tien phai tinh nghich dao cho ma tran A (a1, a2 = a3 = 0, a4) va B (b1, b2)
	double a1_ = 1/a1;		double a4_ = 1/a4;
	double b1_ = a1_*b1;	double b2_ = a4_*b2;

	// main run
	int step = src1->widthStep, width = src1->width, height = src1->height;
	const uchar* src1Data = (uchar *)src1->imageData;
	const uchar* src2Data = (uchar *)src2->imageData;

	int hDiv2 = height/2;
	int wDiv2 = width/2;

	int count = 0;
	int totalError = 0;

	for (int i = 0; i < height; i++) {
		int newY = i - hDiv2;

		for (int j = 0; j < width; j++) {
			int newX = j - wDiv2;				
			int pos = i*step + j;
			double x = a1_*newX - b1_;		x += wDiv2;
			double y = a4_*newY - b2_;		y += hDiv2;	

			if (0 <= x && x < width - 1 && 0 <= y && y < height - 1) { // minimum neighborhood
				int i_ = int(y);
				int j_ = int(x);
				int des1 = src1Data[i_*step + j_]		- src2Data[pos];	if (des1 < 0) des1 = -des1;
				int des2 = src1Data[i_*step + (j_+1)]	- src2Data[pos];	if (des2 < 0) des2 = -des2;		if (des1 > des2) des1 = des2;
				des2 = src1Data[(i_+1)*step + j_]		- src2Data[pos];	if (des2 < 0) des2 = -des2;		if (des1 > des2) des1 = des2;
				des2 = src1Data[(i_+1)*step + (j_+1)]	- src2Data[pos];	if (des2 < 0) des2 = -des2;		if (des1 > des2) des1 = des2;

				count++;
				totalError += des1;
			}
		}
	}

	return totalError * 1.0 / count;
}


void subtractImgWrtAffine(IplImage* src1, IplImage* src2, double a1, double a4, double b1, double b2, IplImage* des) {
		
	// truoc tien phai tinh nghich dao cho ma tran A (a1, a2 = a3 = 0, a4) va B (b1, b2)
	double a1_ = 1/a1;		double a4_ = 1/a4;
	double b1_ = a1_*b1;	double b2_ = a4_*b2;

	// main run
	int step = src1->widthStep, width = src1->width, height = src1->height;
	const uchar* src1Data = (uchar *)src1->imageData;
	const uchar* src2Data = (uchar *)src2->imageData;
	uchar* desData = (uchar *)des->imageData;

	int hDiv2 = height/2;
	int wDiv2 = width/2;

	for (int i = 0; i < height; i++) {
		int newY = i - hDiv2;

		for (int j = 0; j < width; j++) {
			int newX = j - wDiv2;				
			int pos = i*step + j;
			double x = a1_*newX - b1_;		x += wDiv2;
			double y = a4_*newY - b2_;		y += hDiv2;	

			if (0 <= x && x < width - 1 && 0 <= y && y < height - 1) { // minimum neighborhood
				int i_ = int(y);
				int j_ = int(x);
				int des1 = src1Data[i_*step + j_]		- src2Data[pos];	if (des1 < 0) des1 = -des1;
				int des2 = src1Data[i_*step + (j_+1)]	- src2Data[pos];	if (des2 < 0) des2 = -des2;		if (des1 > des2) des1 = des2;
				des2 = src1Data[(i_+1)*step + j_]		- src2Data[pos];	if (des2 < 0) des2 = -des2;		if (des1 > des2) des1 = des2;
				des2 = src1Data[(i_+1)*step + (j_+1)]	- src2Data[pos];	if (des2 < 0) des2 = -des2;		if (des1 > des2) des1 = des2;
				desData[pos] = 255 - des1;
			} else desData[pos] = 255;
		}
	}
}


// ===========================================================================================================================


void main() {
	float totalTime = 0;
	float totalError = 0;

	FILE *fp;
	char charFn[256];


	for(int i = 1; i < BATCH_N_IMG; i++) {

		// ---------- print the image number ----------
		fp = fopen(RESULT_FILENAME, "a");		fprintf(fp, "%d   ", i);		fclose(fp);


		// ---------- load two images ----------
		sprintf(charFn, BATCH_IMG_FILENAME, i-1);						srcImg0 = cvvLoadImage(charFn);					imW = srcImg0->width;	imH = srcImg0->height;	
		grayImg0 = cvCreateImage(cvSize(imW, imH), IPL_DEPTH_8U, 1);	cvCvtColor(srcImg0, grayImg0, CV_RGB2GRAY);
		
		sprintf(charFn, BATCH_IMG_FILENAME, i);							srcImg1 = cvvLoadImage(charFn);
		grayImg1 = cvCreateImage(cvSize(imW, imH), IPL_DEPTH_8U, 1);	cvCvtColor(srcImg1, grayImg1, CV_RGB2GRAY);	
		
		convertCharImgToIntImg();


		// ---------- RUN ----------		
		//totalTime += runTransEst(img0, img1, imH, imW, TRANS_FROM, TRANS_TO, TRANS_RES);
		//totalTime += runScaleEst(img0, img1, imH, imW, SCALE_FROM, SCALE_TO, SCALE_RES);

		runTransEst(img0, img1, imH, imW, TRANS_FROM, TRANS_TO, TRANS_RES);		int dx = getDx();	int dy = getDy();	// dx, dy are also printed to file
		runScaleEst(img0, img1, imH, imW, SCALE_FROM, SCALE_TO, SCALE_RES);		float sy = getSy();						// sy is also printed to file

		totalTime += runGA(img0, img1, imH, imW, 
							GA_POP_SIZE, GA_PC, GA_N_GEN, 
							dx - GA_TRANS_RANGE, dx + GA_TRANS_RANGE, dy - GA_TRANS_RANGE, dy + GA_TRANS_RANGE,
							sy - GA_SCALE_RANGE, sy + GA_SCALE_RANGE);


		// ---------- print the best parameter set & error ----------
		float bestSx, bestSy, bestDx, bestDy, bestError;
		bestSx = getBestSx();
		bestSy = getBestSy();
		bestDx = getBestDx();
		bestDy = getBestDy();
		bestError = getBestError();
		//bestError = calculateAvgIntensityWrtAffine(grayImg0, grayImg1, double(bestSx), double(bestSy), double(bestDx), double(bestDy));

		totalError += bestError;

		fp = fopen(RESULT_FILENAME, "a");		fprintf(fp, "[FINAL] sx=%1.3f, sy=%1.3f, dx=%2.3f, dy=%2.3f, e=%2.3f \n", bestSx, bestSy, bestDx, bestDy, bestError);		fclose(fp);


		// ----------
		release();
	}


	// write total/average information	
	fp = fopen(RESULT_FILENAME, "a");		fprintf(fp, "avgError=%2.3f, avgTime = %2.3f\n", totalError/(BATCH_N_IMG-1), totalTime/(BATCH_N_IMG-1));		fclose(fp);
}
