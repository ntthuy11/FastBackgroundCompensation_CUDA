
extern "C" float getBestSx();
extern "C" float getBestSy();
extern "C" float getBestDx();
extern "C" float getBestDy();

extern "C" float getBestError();

extern "C" float runGA(int *img0, int *img1, int imH, int imW,
					   /* GA parameters */		int popSize, double pc, int nGen,
					   /* translation range */	int dxRangeFrom, int dxRangeTo, int dyRangeFrom, int dyRangeTo,
					   /* scale range */		double sRangeFrom, double sRangeTo);