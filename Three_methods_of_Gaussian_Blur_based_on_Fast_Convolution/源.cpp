#include<opencv2\opencv.hpp>
#include<iostream>
#include"MyMatrix.h"
#include"testFFT.h"

using namespace std;
using namespace cv;

#define PI 3.1415926535
Matrix getGaussFilter(int kernel_h, int kernel_w, double sigma_h, double sigma_w);
inline bool is_a_ge_zero_and_a_lt_b(int a, int b);
template <typename Dtype>//im2col实现https://blog.csdn.net/u013066730/article/details/86489139
void im2col_cpu(const Dtype* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	Dtype* data_col);
void myGaussianBlur_nature(Matrix &srcIm, Matrix &dstIm, Matrix &C_tmp, int kernel_h, int kernel_w, double sigma_h, double sigma_w);
void myGaussianBlur_GEMM(Matrix &srcIm, Matrix &dstIm, Matrix &im2col_res, int kernel_h, int kernel_w, double sigma_h, double sigma_w);
void myGaussianBlur_FFT(Mat &srcIm, Mat &dstIm, Mat &gaussiankernel, int kernel_h, int kernel_w, double sigma_h, double sigma_w);


int main() {

	int iftest = 0; int height = 128;int width = 128;
	//int iftest = 1; int height = 512; int width = 512;
	//int iftest = 1; int height = 8; int width = 8;

	int channels_in = 1;
	int channels_out = 1;
	int kernel_h = 5;
	int kernel_w = 5;
	int pad_h = kernel_h/2;
	int pad_w = kernel_w/2;
	int stride_h = 1;
	int stride_w = 1;
	int dilation_h = 1;
	int dilation_w = 1;

	double sigma_h = 2.0;
	double sigma_w = 2.0;

	

	Mat srcIm_mat = imread("test2.jpg", 0);//原始图像
	resize(srcIm_mat, srcIm_mat, Size(height, width));
	if (iftest) {
		if (height == width&&width > 8) {
			imshow("srcIm_mat", srcIm_mat);
			imwrite("srcIm_mat.bmp", srcIm_mat);


		}
		else {
			cout << "srcIm_mat: " << endl; cout << srcIm_mat << endl;//for (int i = 0; i < srcIm_mat.rows; i++) { for (int j = 0; j < srcIm_mat.cols; j++) { cout << srcIm_mat.at<uchar>(i, j) << "\t"; }cout << endl; }cout << endl;
		}
	}

	

	Matrix srcIm(height, width);
	for (int i = 0; i < srcIm.rows(); i++) {
		for (int j = 0; j < srcIm.cols(); j++) {
			//srcIm(i, j) = i*srcIm.rows() + j + 1;
			srcIm(i, j) = srcIm_mat.at<uchar>(i,j);
		}
	}
	if (iftest) {
		if (height == width&&width <= 8) {
			cout << "srcIm: " << endl; srcIm.print();
		}
	}


	Matrix dstIm1;
	Matrix C_tmp(2 * pad_h + height, 2 * pad_w + width, 0);
	myGaussianBlur_nature(srcIm, dstIm1, C_tmp,kernel_h, kernel_w, sigma_h, sigma_w);
	if (iftest) {
		if (height == width&&width <= 8) {
			cout << "myGaussianBlur_nature: " << endl; dstIm1.print();
		}
		else {
			Mat tmp(dstIm1.rows(), dstIm1.cols(), CV_64FC1, dstIm1.ptr());
			tmp.convertTo(tmp, CV_8UC1);
			imshow("myGaussianBlur_nature", tmp);
			imwrite("myGaussianBlur_nature.bmp", tmp);
		}
	}

	Matrix dstIm2;
	int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;//计算卷积层输出图像的高
	int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;//计算卷积层输出图像的宽
	Matrix im2col_res = Matrix((channels_in*kernel_h*kernel_w), (output_h*output_w), 0);
	myGaussianBlur_GEMM(srcIm, dstIm2, im2col_res, kernel_h, kernel_w, sigma_h, sigma_w);
	if (iftest) {
		if (height == width&&width <= 8) {
			cout << "myGaussianBlur_GEMM: " << endl; dstIm2.print();
			//cout << "GEMM B matrix: " << endl; im2col_res.print();
		}
		else {
			Mat tmp(dstIm2.rows(), dstIm2.cols(), CV_64FC1, dstIm2.ptr());
			tmp.convertTo(tmp, CV_8UC1);
			imshow("myGaussianBlur_GEMM", tmp);
			imwrite("myGaussianBlur_GEMM.bmp", tmp);
		}
	}


	Mat dstIm_mat3;
	srcIm_mat.convertTo(srcIm_mat, CV_64FC1);
	cv::GaussianBlur(srcIm_mat, dstIm_mat3, Size(kernel_h, kernel_w), sigma_h, sigma_w, cv::BORDER_CONSTANT);
	if (iftest) {
		if (height == width&&width > 8) {
			dstIm_mat3.convertTo(dstIm_mat3, CV_8UC1);
			imshow("GaussianBlur", dstIm_mat3);
			imwrite("GaussianBlur.bmp", dstIm_mat3);
		}
		else {
			cout << "GaussianBlur: " << endl; for (int i = 0; i < dstIm_mat3.rows; i++) { for (int j = 0; j < dstIm_mat3.cols; j++) { cout << setiosflags(ios::fixed) << setprecision(3) << dstIm_mat3.at<double>(i, j) << "\t"; }cout << endl; }cout << endl;
		}
	}
	


	Mat dstIm_mat4;
	Mat gaussiankernel = cv::getGaussianKernel(height, sigma_h);
	gaussiankernel = gaussiankernel * gaussiankernel.t();
	gaussiankernel.convertTo(gaussiankernel, CV_32FC1);
	myGaussianBlur_FFT(srcIm_mat, dstIm_mat4, gaussiankernel, kernel_h, kernel_w, sigma_h, sigma_w);
	if (iftest) {
		if (height == width&&width > 8) {
			dstIm_mat4 *= 255;
			dstIm_mat4.convertTo(dstIm_mat4, CV_8UC1);
			imshow("myGaussianBlur_FFT", dstIm_mat4);
			imwrite("myGaussianBlur_FFT.bmp", dstIm_mat4);
		}
		else {
			cout << "myGaussianBlur_FFT: " << endl; for (int i = 0; i < dstIm_mat4.rows; i++) { for (int j = 0; j < dstIm_mat4.cols; j++) { cout << setiosflags(ios::fixed) << setprecision(2) << dstIm_mat4.at<float>(i, j) * 255 << "\t"; }cout << endl; }cout << endl;
		}
	}

	waitKey(0);

	return 0;
}


Matrix getGaussFilter(int kernel_h, int kernel_w, double sigma_h, double sigma_w) {
	if (kernel_h % 2 == 0 || kernel_w % 2 == 0 || kernel_h <= 0 || kernel_w <= 0) { return Matrix(0, 0); }
	Matrix guasskernel(kernel_h, kernel_h);
	int ch = kernel_h / 2, cw = kernel_w / 2;

	double dSum = 0, dValue, dDis;
	int i, j;
	for (i = 0; i< guasskernel.rows(); ++i)
	{
		for (j = 0; j < guasskernel.cols(); ++j)
		{
			dDis = (i - ch) * (i - ch) + (j - cw) * (j - cw);
			dValue = exp(-dDis / (2 * sigma_h * sigma_w)) / (2 * PI * sigma_h * sigma_w);
			guasskernel(i, j) = dValue;
			dSum += dValue;
		}
	}

	// 归一化
	for (i = 0; i< guasskernel.rows(); ++i)
	{
		for (j = 0; j < guasskernel.cols(); ++j)
		{
			guasskernel(i, j) = guasskernel(i, j) / dSum;
		}
	}
	return guasskernel;
}

//该函数定义是：若a大于0且严格小于b，则返回真，否则返回假，该函数的作用是判断矩阵上某元的输出是否为pad的0。
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {//若a大于等于零或小于b，返回true，否则返回false
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>//im2col实现https://blog.csdn.net/u013066730/article/details/86489139
void im2col_cpu(const Dtype* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	Dtype* data_col) {

	const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;//计算卷积层输出图像的高
	const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;//计算卷积层输出图像的宽
	const int channel_size = height * width;//计算卷积层输入单通道图像的数据容量
											//Dtype* data_col= new Dtype[(channels*kernel_h*kernel_w)*(output_h*output_w)];
											/*第一个for循环表示输出的矩阵通道数和卷积层输入图像通道是一样的，每次处理一个输入通道的信息*/
	int channel, kernel_row, kernel_col, input_row, output_rows, output_cols, input_col, output_col, t1, t2, t3;
	for (channel = channels; channel--; data_im += channel_size) {
		/*第二个和第三个for循环表示了输出单通道矩阵的某一列，同时体现了输出单通道矩阵的行数*/
		for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			t1 = -pad_h + kernel_row;
			for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				input_row = t1;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
							   /*第四个和第五个for循环表示了输出单通道矩阵的某一行，同时体现了输出单通道矩阵的列数*/
				t2 = -pad_w + kernel_col;
				for (output_rows = output_h; output_rows; output_rows--) {
					t3 = input_row*width;
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//如果计算得到的输入图像的行值索引小于零或者大于输入图像的高(该行为pad)
						for (output_cols = output_w; output_cols; output_cols--) {
							data_col++;//那么将该行在输出的矩阵上的位置置为0
						}
					}
					else {
						input_col = t2;//在这里找到卷积核中的某一列在输入图像中的第一个操作区域的列索引
									   //index = input_row * width + input_col;
						for (output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//如果计算得到的输入图像的列值索引大于等于于零或者小于输入图像的宽(该列不是pad)
								*(data_col++) = *(data_im + t3 + input_col);//将输入特征图上对应的区域放到输出矩阵上
							}
							else {//否则，计算得到的输入图像的列值索引小于零或者大于输入图像的宽(该列为pad)
								data_col++;//将该行该列在输出矩阵上的位置置为0
							}
							input_col += stride_w;//按照宽方向步长遍历卷积核上固定列在输入图像上滑动操作的区域
						}
					}
					input_row += stride_h;//按照高方向步长遍历卷积核上固定行在输入图像上滑动操作的区域
				}
			}
		}
	}

}


void myGaussianBlur_nature(Matrix &srcIm, Matrix &dstIm, Matrix &C_tmp, int kernel_h, int kernel_w, double sigma_h, double sigma_w) {
	Matrix guasskernel = getGaussFilter(kernel_h, kernel_w, sigma_h, sigma_w);
	//guasskernel.print();
	dstIm = Matrix::conv_nature(srcIm, guasskernel, C_tmp);
	//dstIm.print();
}




void myGaussianBlur_GEMM(Matrix &srcIm, Matrix &dstIm, Matrix &im2col_res, int kernel_h, int kernel_w, double sigma_h, double sigma_w) {

	int height = srcIm.rows();
	int width = srcIm.cols();
	int channels_in = 1;
	int channels_out = 1;
	int pad_h = kernel_h / 2;
	int pad_w = kernel_w / 2;
	int dilation_h = 1;
	int dilation_w = 1;
	int stride_h = 1;
	int stride_w = 1;

	Matrix gausskernel = getGaussFilter(kernel_h, kernel_w, sigma_h, sigma_w);
	gausskernel.flatten();
	//cout << gausskernel.rows() << " " << gausskernel.cols() << endl;
	//gausskernel.print();


	double *data_im = srcIm.ptr();
	int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;//计算卷积层输出图像的高
	int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;//计算卷积层输出图像的宽
																 //Matrix im2col_res = Matrix((channels_in*kernel_h*kernel_w), (output_h*output_w), 0);
	double* data_col = im2col_res.ptr();
	im2col_cpu<double>(data_im, channels_in, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col);
	
	//dstIm = Matrix::multi_ijk(gausskernel, im2col_res);
	//dstIm = Matrix::multi_avx(gausskernel, im2col_res);
	//dstIm = Matrix::multi_avx_unrollx2(gausskernel, im2col_res);
	//dstIm = Matrix::multi_avx_unrollx4(gausskernel, im2col_res);
	dstIm = Matrix::multi_avx_unrollx8_ijk(gausskernel, im2col_res);
	//dstIm = Matrix::multi_avx_unrollx8_jik(gausskernel, im2col_res);
	//dstIm = Matrix::multi_avx_unrollx8_omp(gausskernel, im2col_res);
	//dstIm = Matrix::multi_avx_unrollx16_jik_omp(gausskernel, im2col_res);
	dstIm.reshape(output_h, output_w);
}

void myGaussianBlur_FFT(Mat &srcIm, Mat &dstIm, Mat &gaussiankernel, int kernel_h, int kernel_w, double sigma_h, double sigma_w) {

	int height = srcIm.rows;
	int width = srcIm.cols;
	int channels_in = 1;
	int channels_out = 1;
	int pad_h = kernel_h / 2;
	int pad_w = kernel_w / 2;
	int dilation_h = 1;
	int dilation_w = 1;
	int stride_h = 1;
	int stride_w = 1;

	Mat complexImage;
	FFT(srcIm, complexImage);


	//Mat gaussiankernel = getGaussianKernel(kernel_h, sigma_h);
	//gaussiankernel = gaussiankernel * gaussiankernel.t();//g * g的转置得到二维高斯卷积核
	//gaussiankernel.convertTo(gaussiankernel, CV_32FC1);
	//float sum = 0;
	//for (int i = 0; i < height; i++){
	//	float*p = gaussiankernel.ptr<float>(i);
	//	for (int j = 0; j < height; j++){
	//		float D_pow = pow(i - height / 2, 2) + pow(j - height / 2, 2);
	//		float D_sqrt = sqrtf(D_pow);
	//		if (D_sqrt > kernel_h) {
	//			p[j] = 0;
	//		}
	//		else {
	//			sum += p[j];
	//		}
	//	}
	//}
	//gaussiankernel /= sum;


	Mat complexImagepadded;
	int grows = gaussiankernel.rows;
	int gcols = gaussiankernel.cols;
	float *ptr;
	//int midr = grows / 2;int midc = gcols / 2;float midval = gaussiankernel.at<float>(midr, midc);
	for (int i = 0; i < grows; i++) {
		ptr = gaussiankernel.ptr<float>(i);
		for (int j = 0; j < gcols; j++) {
			//ptr[j] /= midval;
			if ((i + j) & 1) {
				ptr[j] *= -1;
			}
		}
	}

	//Mat padded;
	//int top = (height - kernel_h) / 2, bottom = height - top - kernel_h;
	//int left = (width - kernel_w) / 2, right = width - left - kernel_w;
	////cout << top << " " << bottom << " " << left << " " << right << " " << endl;
	//copyMakeBorder(gaussiankernel, padded, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));
	//padded.convertTo(padded, CV_32FC1);
	//cout << padded.size() << endl;


	Mat planeg[2] = { gaussiankernel ,gaussiankernel };
	merge(planeg, 2, complexImagepadded);
	dft(complexImagepadded, complexImagepadded);//高斯函数的傅里叶变换

	//Mat plane_n[2];
	//split(complexImagepadded, plane_n);
	//normalize(plane_n[0], plane_n[0], 1, 0, NORM_L1);
	//normalize(plane_n[1], plane_n[1], 1, 0, NORM_L1);
	//merge(plane_n, 2, complexImagepadded);

	Mat	convolImage;
	multiply(complexImage, complexImagepadded, convolImage);//频域上，高斯函数的傅里叶变换和图像的傅里叶变换相乘

	Mat shiftedfilteredImage;//频域上的复变函数，需要转换为实函数
	idft(convolImage, shiftedfilteredImage);//其结果进行逆变换，

	
	complex2fftImage(shiftedfilteredImage, dstIm, 0);//得到高斯滤波处理后的图像


	//调整象限
	int cx = dstIm.cols / 2;
	int cy = dstIm.rows / 2;
	Mat q0(dstIm, Rect(0, 0, cx, cy));
	Mat q1(dstIm, Rect(cx, 0, cx, cy));
	Mat q2(dstIm, Rect(0, cy, cx, cy));
	Mat q3(dstIm, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

}

