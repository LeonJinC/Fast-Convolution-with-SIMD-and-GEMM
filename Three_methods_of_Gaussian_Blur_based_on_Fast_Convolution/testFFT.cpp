#include"testFFT.h"
using namespace std;
using namespace cv;



void FFT(Mat& image, Mat &complexImg) {//傅里叶变换
	/*
	图像获取复数
	Mat complexImg;
	FFT(invertedImage, complexImg);
	*/


	Mat padded;//如果图像的尺寸不是2的倍数则要用copyMakeBorder进行填充
	//int w = getOptimalDFTSize(image.cols);int h = getOptimalDFTSize(image.rows);cout << w << " " << h << endl;
	//copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));
	image.convertTo(padded, CV_32FC1);

	shift(padded);//为频谱中心化做的图像预处理
	Mat plane[] = { padded, Mat::zeros(padded.size(), CV_32F) };

	merge(plane, 2, complexImg);
	cv::dft(complexImg, complexImg);//调用opencv的dft函数，自己实现的版本还待优化

}

void complex2fftImage(Mat &complexImage, Mat &fftImage, int alpha){//复数域的傅里叶复变函数，转换为，实数域的幅值函数
	/*
	//Mat fftImage;
	//complex2fftImage(complexImage, fftImage,1);
	//Mat bandrejectfftImage;
	//complex2fftImage(bandreject, bandrejectfftImage,1);
	//Mat convolfftImage;
	//complex2fftImage(convolImage, convolfftImage, 1);
	Mat bandrejectedImage;
	complex2fftImage(shiftedfilteredImage, bandrejectedImage, 0);
	*/
	Mat plane[2];
	split(complexImage, plane);
	calculateSnP(plane[0], plane[1]);
	if (alpha == 1)
	{
		//log(x+1)+normalize
		logImage(plane[0], plane[0]);
	}
	else
	{
		normalize(plane[0], plane[0], 1, 0, CV_MINMAX);
	}

	fftImage = plane[0];
}

void calculateSnP(Mat &Real, Mat &Imaginary){ //计算幅值和相位（为了节省时间，相位没有用到就被我注释了）
	/*
  实数和虚数转化为 未归一化和log化的频谱和相位谱
  */
	
	//flip(Real, Real, 0);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	//flip(Real, Real, 1);
	//Mat tmp = Real.clone();
	flip(Imaginary, Imaginary, 0);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	flip(Imaginary, Imaginary, 1);
	magnitude(Real, Imaginary, Real);
	//phase(tmp, Imaginary, Imaginary);//计算相位（为了节省时间，相位没有用到就被我注释了）
}

void logImage(Mat input, Mat &output){//log处理和归一化处理

	input += Scalar::all(1);
	log(input, input);
	normalize(input, output, 1, 0, CV_MINMAX);

}

void shift(Mat &padded) {//为频谱中心化做的图像预处理
	for (int i = 0; i < padded.rows; i++) {
		float *ptr = padded.ptr<float>(i);
		for (int j = 0; j < padded.cols; j++) {
			if ((i + j) & 1) {
				ptr[j] *= -1;
			}
		}
	}
}

//Mat getPaddedImage(Mat &image)
//{
//	/*
//	Mat padded = getPaddedImage(invertedImage);
//	Mat plane[] = { padded, Mat::zeros(padded.size(), CV_32F) };
//	*/
//	int w = getOptimalDFTSize(image.cols);
//	int h = getOptimalDFTSize(image.rows);
//	Mat padded;
//	copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));
//	padded.convertTo(padded, CV_32FC1);
//	return padded;
//}
//
//void getBandPass(Mat &padded, Mat &bandpass)
//{
//
//	/*
//	Mat padded = getPaddedImage(invertedImage);
//	Mat bandpass(padded.size(), CV_32FC2);//两通道
//	*/
//	float D0 = 1;
//	float W = 100;
//	for (int i = 0; i < padded.rows; i++)
//	{
//		float*p = bandpass.ptr<float>(i);
//
//		for (int j = 0; j < padded.cols; j++)
//		{
//			float D_pow = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
//			float D_sqrt = sqrtf(D_pow);
//			float D0_pow = pow(D0, 2);
//			p[2 * j] = expf(-pow(((D_pow - D0_pow) / (D_sqrt*W)), 2));
//			p[2 * j + 1] = expf(-pow(((D_pow - D0_pow) / (D_sqrt*W)), 2));
//			//p[2 * j] = 1;
//			//p[2 * j + 1] = 1;
//		}
//	}
//
//}
//
//void getBandreject(Mat &padded, Mat &bandpass)
//{
//
//	/*
//	Mat padded = getPaddedImage(invertedImage);
//	Mat bandreject(padded.size(), CV_32FC2);//两通道
//	*/
//	float D0 = 3;
//	float W = 10;
//	for (int i = 0; i < padded.rows; i++)
//	{
//		float*p = bandpass.ptr<float>(i);
//
//		for (int j = 0; j < padded.cols; j++)
//		{
//			float D_pow = pow(i - padded.rows / 2, 2) + pow(j - padded.cols / 2, 2);
//			float D_sqrt = sqrtf(D_pow);
//			float D0_pow = pow(D0, 2);
//			p[2 * j] = 1 - expf(-pow(((D_pow - D0_pow) / (D_sqrt*W)), 2));
//			p[2 * j + 1] = 1 - expf(-pow(((D_pow - D0_pow) / (D_sqrt*W)), 2));
//		}
//	}
//
//}
