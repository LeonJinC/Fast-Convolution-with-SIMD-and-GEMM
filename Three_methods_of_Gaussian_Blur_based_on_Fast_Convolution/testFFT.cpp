#include"testFFT.h"
using namespace std;
using namespace cv;



void FFT(Mat& image, Mat &complexImg) {//����Ҷ�任
	/*
	ͼ���ȡ����
	Mat complexImg;
	FFT(invertedImage, complexImg);
	*/


	Mat padded;//���ͼ��ĳߴ粻��2�ı�����Ҫ��copyMakeBorder�������
	//int w = getOptimalDFTSize(image.cols);int h = getOptimalDFTSize(image.rows);cout << w << " " << h << endl;
	//copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));
	image.convertTo(padded, CV_32FC1);

	shift(padded);//ΪƵ�����Ļ�����ͼ��Ԥ����
	Mat plane[] = { padded, Mat::zeros(padded.size(), CV_32F) };

	merge(plane, 2, complexImg);
	cv::dft(complexImg, complexImg);//����opencv��dft�������Լ�ʵ�ֵİ汾�����Ż�

}

void complex2fftImage(Mat &complexImage, Mat &fftImage, int alpha){//������ĸ���Ҷ���亯����ת��Ϊ��ʵ����ķ�ֵ����
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

void calculateSnP(Mat &Real, Mat &Imaginary){ //�����ֵ����λ��Ϊ�˽�ʡʱ�䣬��λû���õ��ͱ���ע���ˣ�
	/*
  ʵ��������ת��Ϊ δ��һ����log����Ƶ�׺���λ��
  */
	
	//flip(Real, Real, 0);// ��תģʽ��flipCode == 0��ֱ��ת����X�ᷭת����flipCode>0ˮƽ��ת����Y�ᷭת����flipCode<0ˮƽ��ֱ��ת������X�ᷭת������Y�ᷭת���ȼ�����ת180�㣩
	//flip(Real, Real, 1);
	//Mat tmp = Real.clone();
	flip(Imaginary, Imaginary, 0);// ��תģʽ��flipCode == 0��ֱ��ת����X�ᷭת����flipCode>0ˮƽ��ת����Y�ᷭת����flipCode<0ˮƽ��ֱ��ת������X�ᷭת������Y�ᷭת���ȼ�����ת180�㣩
	flip(Imaginary, Imaginary, 1);
	magnitude(Real, Imaginary, Real);
	//phase(tmp, Imaginary, Imaginary);//������λ��Ϊ�˽�ʡʱ�䣬��λû���õ��ͱ���ע���ˣ�
}

void logImage(Mat input, Mat &output){//log����͹�һ������

	input += Scalar::all(1);
	log(input, input);
	normalize(input, output, 1, 0, CV_MINMAX);

}

void shift(Mat &padded) {//ΪƵ�����Ļ�����ͼ��Ԥ����
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
//	Mat bandpass(padded.size(), CV_32FC2);//��ͨ��
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
//	Mat bandreject(padded.size(), CV_32FC2);//��ͨ��
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
