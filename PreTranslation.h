
extern "C" int getDx();
extern "C" int getDy();

extern "C" float runTransEst(int *img0, int *img1, int imH, int imW,
							 int transFrom, int transTo, int transRes);