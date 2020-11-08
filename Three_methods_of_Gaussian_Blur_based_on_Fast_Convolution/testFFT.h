#include<opencv2/opencv.hpp>
#include<iostream>



void FFT(cv::Mat& image, cv::Mat &complexImg);//傅里叶变换
void complex2fftImage(cv::Mat &complexImage, cv::Mat &fftImage, int alpha);//复数域的傅里叶复变函数，转换为，实数域的幅值函数
void calculateSnP(cv::Mat &Real, cv::Mat &Imaginary);//计算幅值和相位（为了节省时间，相位没有用到就被我注释了）
void logImage(cv::Mat input, cv::Mat &output);//log处理和归一化处理
void shift(cv::Mat &padded);//为频谱中心化做的图像预处理

//cv::Mat getPaddedImage(cv::Mat &image);
//void getBandPass(cv::Mat &padded, cv::Mat &bandpass);
//void getBandreject(cv::Mat &padded, cv::Mat &bandpass);