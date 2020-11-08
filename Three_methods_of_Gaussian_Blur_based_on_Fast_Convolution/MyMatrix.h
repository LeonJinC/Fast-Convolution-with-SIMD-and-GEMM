#ifndef MYMATRIX_H
#define MYMATRIX_H

#include<iostream>
#include<iomanip>
#include<time.h>
#include<immintrin.h>
#include<omp.h>
#include<thread>
#include<future>


class Matrix {//�����ȴ洢
private:
	double **_Matrix;
	double *_row_Matrix;
	size_t _Row, _Column;
public:
	Matrix() :_Matrix(nullptr), _Row(0), _Column(0) {}//Ĭ�Ϲ���
	Matrix(size_t r, size_t c);//����r�С�c�еľ���
	Matrix(size_t r, size_t c, const double init);//����r�С�c�еľ�����init��ʼ��
	Matrix(const Matrix& B);//��������
	~Matrix();//��������
	//double& operator()(size_t i, size_t j) { return _Matrix[i][j]; }//���ʵ�i�С���j�е�Ԫ��
	//const double operator()(size_t i, size_t j)const { return _Matrix[i][j]; }//���ʵ�i�С���j�е�Ԫ��
	//double** ptr() { return _Matrix; }//���صײ��ά�����ָ�룬��ָ��Ϊ�����������ָ��

	double& operator()(size_t i, size_t j) { return _row_Matrix[i*_Column + j]; }//���ʵ�i�С���j�е�Ԫ��
	const double operator()(size_t i, size_t j)const { return _row_Matrix[i*_Column + j]; }//���ʵ�i�С���j�е�Ԫ��
	double& operator[](size_t i) { return _row_Matrix[i]; }
	const double operator[](size_t i)const { return _row_Matrix[i]; }
	double* ptr() { return _row_Matrix; }//���صײ��ά�����ָ�룬��ָ��Ϊ�����������ָ��

	Matrix& operator=(Matrix&& B);//�ƶ�������ֵ
	int rows() {return _Row;}
	int cols() {return _Column;}
	void flatten();//��ά������������������
	void reshape(size_t rows, size_t cols);
	void print();//��ӡ������Ϣ

	//GEMM�Ķ��ʵ��
	static Matrix multi_ijk(Matrix & A, Matrix & B);
	static Matrix multi_avx(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx2(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx4(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx8_ijk(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx8_jik(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx8_omp(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx16_jik_omp(Matrix & A, Matrix & B);

	static Matrix conv_nature(Matrix & A, Matrix & B, Matrix &C_tmp);//Ĭ�����ĳ���������
	static void padding(Matrix & A, int pad_h, int pad_w,Matrix& C);//�������Ե���Ա߽�Ϊ�Գ��ᷴ�临�����أ��ο�opencv��cv::BorderTypes��BORDER_REFLECT_101
};



#endif // !MYMATRIX_H
