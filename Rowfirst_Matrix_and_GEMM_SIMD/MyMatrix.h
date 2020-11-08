#ifndef MYMATRIX_H
#define MYMATRIX_H

#include<iostream>
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
	double& operator()(size_t i, size_t j) { return _row_Matrix[i*_Column+j]; }//���ʵ�i�С���j�е�Ԫ��
	const double operator()(size_t i, size_t j)const { return _row_Matrix[i*_Column + j]; }//���ʵ�i�С���j�е�Ԫ��
	double& operator[](size_t i) { return _row_Matrix[i]; }
	const double operator[](size_t i)const { return _row_Matrix[i]; }
	double* ptr() { return _row_Matrix; }//���صײ��ά�����ָ�룬��ָ��Ϊ�����������ָ��
	Matrix& operator=(Matrix&& B);//�ƶ�������ֵ
	int rows() {return _Row;}
	int cols() {return _Column;}
	//Matrix flatten();//��ά������������������
	void print();//��ӡ������Ϣ

	//GEMM�Ķ��ʵ��
	static Matrix multi(Matrix & A, Matrix & B);
	static Matrix multi_midk(Matrix & A, Matrix & B);
	static Matrix multi_register(Matrix & A, Matrix & B);
	static Matrix multi_loopunrolling(Matrix & A, Matrix & B);
	static Matrix multi_avx(Matrix & A, Matrix & B);
	static Matrix multi_avx_blk(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx2(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx4(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx4_blk(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx4_blk_omp(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx8_ijk(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx8_jik(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx8_omp(Matrix & A, Matrix & B);
	static Matrix multi_avx_unrollx16_jik_omp(Matrix & A, Matrix & B);
	//static void loopunrolling_1x4packing_kernel(Matrix & A0, Matrix & B0, Matrix & C0, int row, int col);
	//static Matrix multi_loopunrolling_1x4packing(Matrix & A, Matrix & B);
	//static void loopunrolling_packing_1x4simd128x1_kernel(Matrix & A0, Matrix & B0, Matrix & C0, int row, int col);
	//static Matrix multi_loopunrolling_packing_1x4simd128x1(Matrix & A, Matrix & B);
	//static void loopunrolling_packing_1x4simd256x1_kernel(Matrix & A0, Matrix & B0, Matrix & C0, int row, int col);
	//static Matrix multi_loopunrolling_packing_1x4simd256x1(Matrix & A, Matrix & B);
	//static void loopunrolling_packing_1x8simd256x2_kernel(Matrix & A0, Matrix & B0, Matrix & C0, int row, int col);
	//static Matrix multi_loopunrolling_packing_1x8simd256x2(Matrix & A, Matrix & B);
	//static Matrix multi_loopunrolling_4x4register(Matrix & A, Matrix & B);
	//static void loopunrolling_packing_4x8simd256x2_kernel(Matrix & A0, Matrix & B0, Matrix & C0, int row, int col);
	//static Matrix multi_loopunrolling_packing_4x8simd256x2(Matrix & A, Matrix & B);

	//
	//static Matrix conv_nature(Matrix & A, Matrix & B);//Ĭ�����ĳ���������
	//static Matrix padding(Matrix & A, int pad_h, int pad_w);//�������Ե���Ա߽�Ϊ�Գ��ᷴ�临�����أ��ο�opencv��cv::BorderTypes��BORDER_REFLECT_101
	//
};



#endif // !MYMATRIX_H
