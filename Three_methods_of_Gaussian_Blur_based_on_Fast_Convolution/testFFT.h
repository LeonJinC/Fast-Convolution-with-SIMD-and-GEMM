#include<opencv2/opencv.hpp>
#include<iostream>



void FFT(cv::Mat& image, cv::Mat &complexImg);//����Ҷ�任
void complex2fftImage(cv::Mat &complexImage, cv::Mat &fftImage, int alpha);//������ĸ���Ҷ���亯����ת��Ϊ��ʵ����ķ�ֵ����
void calculateSnP(cv::Mat &Real, cv::Mat &Imaginary);//�����ֵ����λ��Ϊ�˽�ʡʱ�䣬��λû���õ��ͱ���ע���ˣ�
void logImage(cv::Mat input, cv::Mat &output);//log����͹�һ������
void shift(cv::Mat &padded);//ΪƵ�����Ļ�����ͼ��Ԥ����

//cv::Mat getPaddedImage(cv::Mat &image);
//void getBandPass(cv::Mat &padded, cv::Mat &bandpass);
//void getBandreject(cv::Mat &padded, cv::Mat &bandpass);