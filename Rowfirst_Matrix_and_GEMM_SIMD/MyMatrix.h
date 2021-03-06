#ifndef MYMATRIX_H
#define MYMATRIX_H

#include<iostream>
#include<time.h>
#include<immintrin.h>
#include<omp.h>
#include<thread>
#include<future>

class Matrix {//行优先存储
private:
	double **_Matrix;
	double *_row_Matrix;
	size_t _Row, _Column;
public:
	Matrix() :_Matrix(nullptr), _Row(0), _Column(0) {}//默认构造
	Matrix(size_t r, size_t c);//构造r行、c列的矩阵
	Matrix(size_t r, size_t c, const double init);//构造r行、c列的矩阵并用init初始化
	Matrix(const Matrix& B);//拷贝构造
	~Matrix();//析构函数
	double& operator()(size_t i, size_t j) { return _row_Matrix[i*_Column+j]; }//访问第i行、第j列的元素
	const double operator()(size_t i, size_t j)const { return _row_Matrix[i*_Column + j]; }//访问第i行、第j列的元素
	double& operator[](size_t i) { return _row_Matrix[i]; }
	const double operator[](size_t i)const { return _row_Matrix[i]; }
	double* ptr() { return _row_Matrix; }//返回底层二维数组的指针，该指针为矩阵的首行行指针
	Matrix& operator=(Matrix&& B);//移动拷贝赋值
	int rows() {return _Row;}
	int cols() {return _Column;}
	//Matrix flatten();//二维矩阵向量化，行优先
	void print();//打印矩阵信息

	//GEMM的多个实现
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
	//static Matrix conv_nature(Matrix & A, Matrix & B);//默认填充的常规卷积操作
	//static Matrix padding(Matrix & A, int pad_h, int pad_w);//填充矩阵边缘，以边界为对称轴反射复制像素，参考opencv的cv::BorderTypes的BORDER_REFLECT_101
	//
};



#endif // !MYMATRIX_H
