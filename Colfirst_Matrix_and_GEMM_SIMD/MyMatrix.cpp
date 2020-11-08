#include"MyMatrix.h"
using namespace std;

void* aligned_malloc(size_t size, int alignment)
{
	// �����㹻���ڴ�, ������㷨�ܾ���, ���ڵ�STL��ʹ�õľ�������㷨  

	// ������ά��FreeBlockָ��ռ�õ��ڴ��С  
	const int pointerSize = sizeof(void*);

	// alignment - 1 + pointerSize�����FreeBlock�ڴ������Ҫ���ڴ��С  
	// ǰ�������sizeof(T) = 20, __alignof(T) = 16,  
	// g_MaxNumberOfObjectsInPool = 1000  
	// ��ô���ñ���������alignedMalloc(1000 * 20, 16)  
	// ��ôalignment - 1 + pointSize = 19  
	const int requestedSize = size + alignment - 1 + pointerSize;

	// �����ʵ�ʴ�С����20000 + 19 = 20019  
	void* raw = malloc(requestedSize);

	// ����ʵPool����Ϊ����ʵ��������ڴ��ַ  
	uintptr_t start = (uintptr_t)raw + pointerSize;
	// �����������  
	// ����һ��, __ALIGN - 1ָ������ʵ���ڴ���������  
	// ����__ALIGN = 8ʱ, ����ֻ��Ҫ7�Ϳ���ʵ�ʱ�ʾ8����(0~7)  
	// ��ô~(__ALIGN - 1)���ǽ������������  
	// ���ǽ�(bytes) + __ALIGN-1)�����Ƚ��н�λ, Ȼ��ض�  
	// ��ͱ�֤���������������  
	// ����byte = 100, __ALIGN = 8�����  
	// ~(__ALIGN - 1) = (1 000)B  
	// ((bytes) + __ALIGN-1) = (1 101 011)B  
	// (((bytes) + __ALIGN-1) & ~(__ALIGN - 1)) = (1 101 000 )B = (104)D  
	// 104 / 8 = 13, ���ʵ������������  
	// ����byte�պ������ڴ����������, �������byte��С����  
	// �ǵá�Hacker's Delight����������صļ���  
	// ������ʽ����������ĵȼ�  
	// ((((bytes) + _ALIGN - 1) * _ALIGN) / _ALIGN)  
	// ����SGI STLʹ�õķ���Ч�ʷǳ���   
	void* aligned = (void*)((start + alignment - 1) & ~(alignment - 1));

	// ����ά��һ��ָ��malloc()����������ڴ�  
	*(void**)((uintptr_t)aligned - pointerSize) = raw;

	// ����ʵ�����������ĵ�ַ  
	return aligned;
}


// �������ڲ�ά�����ڴ����  
//                   ���������ڴ����Ҫ��  
//                             |  
// ----------------------------------------------------------------------  
// | �ڴ������� | ά����ָ�� | ����1 | ����2 | ����3 | ...... | ����n |  
// ----------------------------------------------------------------------  
// ^                     | ָ��malloc()����ĵ�ַ���  
// |                     |  
// -----------------------  
void aligned_free(void * aligned_ptr)
{
	if (aligned_ptr)
	{
		free(((void**)aligned_ptr)[-1]);
	}
}

bool isAligned(void* data, int alignment)
{
	// ����һ�������㷨, �μ�<Hacker's Delight>  
	return ((uintptr_t)data & (alignment - 1)) == 0;
}

Matrix::Matrix(size_t r, size_t c) :_Row(r), _Column(c) {//����r�С�c�еľ���
	if (!_Column || !_Row) return;
	//_row_Matrix = new double[_Row*_Column];
	_row_Matrix = (double*)aligned_malloc(sizeof(double)*_Row*_Column,32);
	//std::cout << "address of _row_Matrix: " << _row_Matrix << std::endl;
	//if (isAligned(_row_Matrix, 32)) {
	//	cout << "isAligned" << endl;
	//}
	//else {
	//	cout << "not Aligned" << endl;
	//}
}
Matrix::Matrix(size_t r, size_t c, const double init) :_Row(r), _Column(c) {//����r�С�c�еľ�����init��ʼ��
	if (!_Column || !_Row) return;
	//_row_Matrix = new double[_Row*_Column]; 
	_row_Matrix = (double*)aligned_malloc(sizeof(double)*_Row*_Column, 32);
	//std::cout << "address of _row_Matrix: " << _row_Matrix << std::endl;
	//if (isAligned(_row_Matrix, 32)) {
	//	cout << "isAligned" << endl;
	//}
	//else {
	//	cout << "not Aligned" << endl;
	//}
	memset(_row_Matrix, init, _Row*_Column*sizeof(double));
}
Matrix::Matrix(const Matrix& B) {//��������
						 //cout << "��������" << endl;
	_Row = B._Row;
	_Column = B._Column;
	//_row_Matrix = new double[_Row*_Column]; 
	_row_Matrix = (double*)aligned_malloc(sizeof(double)*_Row*_Column, 32);
	//std::cout << "address of _row_Matrix: " << _row_Matrix << std::endl;
	//if (isAligned(_row_Matrix, 32)) {
	//	cout << "isAligned" << endl;
	//}
	//else {
	//	cout << "not Aligned" << endl;
	//}
	memmove(_row_Matrix, B._row_Matrix, _Row*_Column * sizeof(double));

}
Matrix::~Matrix() {//��������
	if (!_Matrix) return;
	//double **p = _Matrix, **end = _Matrix + _Row;
	//do {
	//	delete[](*(p++));
	//} while (p != end);
	//delete[] _Matrix;
	//delete[] _row_Matrix;
	aligned_free(_row_Matrix);
	_Column = _Row = 0;
	//cout << "��������" << endl;
}

Matrix& Matrix::operator=(Matrix&& B) {//�ƶ�������ֵ
							   //cout << "�ƶ���ֵ" << endl;
	//if (_Matrix) {
	//	double **p = _Matrix, **end = _Matrix + _Row;
	//	do {
	//		delete[](*(p++));
	//	} while (p != end);
	//	delete[] _Matrix;
	//}
	//_Row = B._Row;
	//_Column = B._Column;
	//_Matrix = B._Matrix;
	//B._Matrix = nullptr;
	//return *this;

	if (_row_Matrix) {
		//delete[] _row_Matrix;
		aligned_free(_row_Matrix);
	}
	_Row = B._Row;
	_Column = B._Column;
	_row_Matrix = B._row_Matrix;
	B._row_Matrix = nullptr;
	return *this;
}

//Matrix Matrix::flatten() {
	//Matrix flat(1, _Row*_Column);
	//for (int i = 0; i < _Row; i++) {
	//	for (int j = 0; j < _Column; j++) {
	//		flat(0, i * _Row + j) = _Matrix[i][j];
	//	}
	//}
	//return flat;
//}

void Matrix::print() {
	//for (size_t i = 0; i < _Row; i++) {
	//	for (size_t j = 0; j < _Column; j++) {
	//		std::cout << _Matrix[i][j] << "\t";
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;
	for (size_t i = 0; i < _Row; i++) {
		for (size_t j = 0; j < _Column; j++) {
			std::cout << _row_Matrix[i+j*_Row] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	//for (size_t i = 0; i < _Row*_Column; i++) {
	//	std::cout << _row_Matrix[i] << "\t";
	//}
	//std::cout << std::endl;

}

Matrix Matrix::multi(Matrix &A, Matrix &B) {
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);

	size_t Crows=C.rows(),Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	
	for (size_t j = 0; j < Ccols; j++) {
		for (size_t i = 0; i < Crows; i++) {
			for (size_t k = 0; k < Acols; k++) {
				C[i +j*Crows] += A[i + k*Arows] * B[k + j*Brows];
			}
		}
	}
	//C.print();
	return C;
}

Matrix Matrix::multi_midk(Matrix &A, Matrix &B) {
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	for (size_t j = 0; j < Ccols; j++) {
		for (size_t k = 0; k < Acols; k++) {
			for (size_t i = 0; i < Crows; i++) {
				C[i + j*Crows] += A[i + k*Arows] * B[k + j*Brows];
			}
		}
	}
	return C;
}

Matrix Matrix::multi_register(Matrix &A, Matrix &B) {
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);

	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	register double temp;
	for (size_t j = 0; j < Ccols; j++) {
		for (size_t i = 0; i < Crows; i++) {
			temp = 0;
			for (size_t k = 0; k < Acols; k++) {
				temp += A[i + k*Arows] * B[k + j*Brows];
			}
			C[i + j*Crows] = temp;
		}
	}
	return C;

}

Matrix Matrix::multi_loopunrolling(Matrix &A, Matrix &B) {
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	register double t0, t1, t2, t3,temp;
	for (size_t j = 0; j < Ccols; j += 4) {
		for (size_t i = 0; i < Crows; i++) {
			//double t0(0), t1(0), t2(0), t3(0);
			t0 = 0; t1 = 0; t2 = 0; t3 = 0;
			for (size_t k = 0; k < Acols; k++) {
				temp = A[i + k*Arows];
				t0 += temp * B[k + (j + 0)*Brows];
				t1 += temp * B[k + (j + 1)*Brows];
				t2 += temp * B[k + (j + 2)*Brows];
				t3 += temp * B[k + (j + 3)*Brows];
			}
			C[i + (j + 0)*Crows] = t0;
			C[i + (j + 1)*Crows] = t1;
			C[i + (j + 2)*Crows] = t2;
			C[i + (j + 3)*Crows] = t3;
		}
	}
	return C;
}

Matrix Matrix::multi_avx(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	double  *c = C.ptr();
	double  *b = B.ptr();
	double  *a = A.ptr();

	__m256d a0, b0, c0;
	for (size_t j = 0; j < Ccols; j++) {
		for (size_t i = 0; i < Crows; i+=4) {
			c0 = _mm256_load_pd(c + i + j*Crows); /* c0 = C[i][j] */
			for (size_t k = 0; k < Acols; k++) {
				a0 = _mm256_load_pd(a + i + k*Arows);
				b0 = _mm256_broadcast_sd(b + k + j*Brows);
				c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
			}
			_mm256_store_pd(c + i + j*Crows, c0);  /* C[i][j] = c0 */;
		}
	}
	return C;
}

#define UNROLL 4
Matrix Matrix::multi_avx_unrollx4(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	double  *c = C.ptr();
	double  *b = B.ptr();
	double  *a = A.ptr();
	__m256d cm[UNROLL],b0;
	for (size_t j = 0; j < Ccols; j++) {
		for (size_t i = 0; i < Crows; i += 4 * UNROLL) {
		
			//for (int x = 0; x < UNROLL; x++) {
			//	cm[x] = _mm256_load_pd(c + i + x * 4 + j*Crows);
			//}
			cm[0] = _mm256_load_pd(c + i + j*Crows);
			cm[1] = _mm256_load_pd(c + i + 4 + j*Crows);
			cm[2] = _mm256_load_pd(c + i + 8 + j*Crows);
			cm[3] = _mm256_load_pd(c + i + 12 + j*Crows);
			for (size_t k = 0; k < Acols; k++) {
				b0 = _mm256_broadcast_sd(b + k + j*Brows);
				//for (int x = 0; x < UNROLL; x++) {
				//	cm[x] = _mm256_add_pd(cm[x],_mm256_mul_pd(_mm256_load_pd(a + i + k*Arows + x * 4),b0));
				//}
				cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(_mm256_load_pd(a + i + k*Arows), b0));
				cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(_mm256_load_pd(a + i + k*Arows + 4), b0));
				cm[2] = _mm256_add_pd(cm[2], _mm256_mul_pd(_mm256_load_pd(a + i + k*Arows + 8), b0));
				cm[3] = _mm256_add_pd(cm[3], _mm256_mul_pd(_mm256_load_pd(a + i + k*Arows + 12), b0));
			}
			//for (int x = 0; x < UNROLL; x++) {
			//	_mm256_store_pd(c + i + x * 4 + j*Crows, cm[x]);
			//}
			_mm256_store_pd(c + i + j*Crows, cm[0]);
			_mm256_store_pd(c + i + 4 + j*Crows, cm[1]);
			_mm256_store_pd(c + i + 8 + j*Crows, cm[2]);
			_mm256_store_pd(c + i + 12 + j*Crows, cm[3]);
		}
	}
	return C;
}


#define BLOCKSIZE 32
Matrix Matrix::multi_avx_unrollx4_blk(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	double  *c = C.ptr();
	double  *b = B.ptr();
	double  *a = A.ptr();
	for (int sj = 0; sj < Ccols; sj += BLOCKSIZE) {
    for (int sk = 0; sk < Acols; sk += BLOCKSIZE) {
      for (int si = 0; si < Crows; si += BLOCKSIZE) {
			
				for (int j = sj; j < sj + BLOCKSIZE; j++) {
					for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4) {
						__m256d cm[UNROLL];
						for (int x = 0; x < UNROLL; x++) {
							cm[x] = _mm256_load_pd(c + i + x * 4 + j*Crows);
						}
						for (int k = sk; k < sk + BLOCKSIZE; k++) {
							__m256d b0 = _mm256_broadcast_sd(b + k + j*Brows);
							for (int x = 0; x < UNROLL; x++) {
								cm[x] = _mm256_add_pd(cm[x],
									_mm256_mul_pd(
										_mm256_load_pd(a + Arows*k + x * 4 + i), b0));
							}
						}

						for (int x = 0; x < UNROLL; x++) {
							_mm256_store_pd(c + i + x * 4 + j*Crows, cm[x]);
						}
					}
				}
			}
		}
	}
	return C;
}

Matrix Matrix::multi_avx_unrollx4_blk_omp(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	double  *c = C.ptr();
	double  *b = B.ptr();
	double  *a = A.ptr();
#pragma omp parallel for
	for (int sj = 0; sj < Ccols; sj += BLOCKSIZE) {
    for (int sk = 0; sk < Acols; sk += BLOCKSIZE) {
      for (int si = 0; si < Crows; si += BLOCKSIZE) {
				for (int j = sj; j < sj + BLOCKSIZE; j++) {
					for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4) {
						__m256d cm[UNROLL];
						for (int x = 0; x < UNROLL; x++) {
							cm[x] = _mm256_load_pd(c + i + x * 4 + j*Crows);
						}
						for (int k = sk; k < sk + BLOCKSIZE; k++) {
							__m256d b0 = _mm256_broadcast_sd(b + k + j*Brows);
							for (int x = 0; x < UNROLL; x++) {
								cm[x] = _mm256_add_pd(cm[x],
									_mm256_mul_pd(
										_mm256_load_pd(a + Arows*k + x * 4 + i), b0));
							}
						}

						for (int x = 0; x < UNROLL; x++) {
							_mm256_store_pd(c + i + x * 4 + j*Crows, cm[x]);
						}
					}
				}
			}
		}
	}
	return C;
}




//void Matrix::loopunrolling_1x4packing_kernel(Matrix &A0, Matrix &B0, Matrix &C0, int row, int col) {
//	double** A(A0.ptr());
//	double** B(B0.ptr());
//	double** C(C0.ptr());
//	double t0(0), t1(0), t2(0), t3(0);
//	double *a0(*(A + row)), *b0(*(B + 0)), *b1(*(B + 1)), *b2(*(B + 2)), *b3(*(B + 3)), *end = b0 + A0.cols();
//	do {
//		t0 += *(a0)**(b0++);
//		t1 += *(a0)**(b1++);
//		t2 += *(a0)**(b2++);
//		t3 += *(a0++)**(b3++);
//	} while (b0 != end);
//	*(*(C + row) + col) = t0;
//	*(*(C + row) + col + 1) = t1;
//	*(*(C + row) + col + 2) = t2;
//	*(*(C + row) + col + 3) = t3;
//}
//Matrix Matrix::multi_loopunrolling_1x4packing(Matrix &A, Matrix &B) {
//
//	if (A.cols() != B.rows()) {
//		return Matrix(0, 0);
//	}
//	Matrix C(A.rows(), B.cols(), 0);
//	Matrix tr(4, A.cols(), 0);
//	int i(0), j(0);
//	do {
//		j = 0;
//		do {
//			tr(0, j) = B(j, i);//packing���̣��������ݴ���������ռ�
//			tr(1, j) = B(j, i + 1);
//			tr(2, j) = B(j, i + 2);
//			tr(3, j) = B(j, i + 3);
//		} while ((++j)<A.cols());
//		j = 0;
//		do {
//			loopunrolling_1x4packing_kernel(A, tr, C, j, i);
//			j += 1;
//		} while (j<C.rows());
//		i += 4;
//	} while (i<C.cols());
//
//	return C;
//}
//
//void Matrix::loopunrolling_packing_1x4simd128x1_kernel(Matrix &A0, Matrix &B0, Matrix &C0, int row, int col) {
//	double** A(A0.ptr());
//	double** B(B0.ptr());
//	double** C(C0.ptr());
//	__m128d t0, t1, a0, b0, b1;
//	t0 = t1 = _mm_set1_pd(0);
//	double *pa0(*(A + row));
//	double *pb0(*(B + 0)), *pb1(*(B + 1));
//	double *end = pa0 + A0.cols();
//	do {
//		b0 = _mm_load_pd(pb0);
//		b1 = _mm_load_pd(pb1);
//		a0 = _mm_set1_pd(*(pa0++));
//
//		t0 = _mm_add_pd(t0, _mm_mul_pd(a0, b0));
//		t1 = _mm_add_pd(t1, _mm_mul_pd(a0, b1));
//		pb0 += 2;
//		pb1 += 2;
//
//	} while (pa0 != end);
//	_mm_store_pd((*(C + row) + col), t0);
//	_mm_store_pd((*(C + row) + col + 2), t1);
//}
//Matrix Matrix::multi_loopunrolling_packing_1x4simd128x1(Matrix &A, Matrix &B) {
//	if (A.cols() != B.rows()) {
//		return Matrix(0, 0);
//	}
//	Matrix C(A.rows(), B.cols(), 0);
//	Matrix tr(2, 2 * A.cols(), 0);
//	int i(0), j(0), k;
//	do {
//		j = 0; k = 0;
//		do {
//			tr(0, k) = B(j, i);//packing���̣��������ݴ���������ռ�
//			tr(1, k++) = B(j, i + 2);
//			tr(0, k) = B(j, i + 1);
//			tr(1, k++) = B(j++, i + 3);
//		} while (j<A.cols());
//		j = 0;
//		do {
//			loopunrolling_packing_1x4simd128x1_kernel(A, tr, C, j, i);
//			j += 1;
//		} while (j<C.rows());
//		i += 4;
//	} while (i<C.cols());
//	return C;
//}
//
//void Matrix::loopunrolling_packing_1x4simd256x1_kernel(Matrix &A0, Matrix &B0, Matrix &C0, int row, int col) {
//	double** A(A0.ptr());
//	double** B(B0.ptr());
//	double** C(C0.ptr());
//	__m256d t0, a0, b0;
//	t0 = _mm256_set1_pd(0);
//
//	double *pa0(A[row]);
//	double *pb0(B[0]);
//	double *end0 = A[row] + A0.cols();
//	do {
//		b0 = _mm256_loadu_pd(pb0);
//		a0 = _mm256_set1_pd(*(pa0++));
//		t0 = _mm256_add_pd(t0, _mm256_mul_pd(a0, b0));
//		pb0 += 4;
//	} while (pa0 != end0);
//
////
////#pragma omp parallel for 
////	for (int i = 0; i < A0.cols(); i++) {
////		double *pa0 = A[row] + i;
////		double *pb0 = B[0] + 4 * i;
////		//b0 = _mm256_loadu_pd(pb0);
////		//a0 = _mm256_set1_pd(*pa0);
////#pragma omp critical
////		t0 = _mm256_add_pd(t0, _mm256_mul_pd(_mm256_set1_pd(*pa0), _mm256_loadu_pd(pb0)));
////	}
//
//	_mm256_storeu_pd(&C[row][col], t0);
//	//cout << "C[" << row << "][" << col << "]: " << C[row][col] << endl;
//	//cout << "C[" << row << "][" << col + 1 << "]: " << C[row][col + 1] << endl;
//	//cout << "C[" << row << "][" << col + 2 << "]: " << C[row][col + 2] << endl;
//	//cout << "C[" << row << "][" << col + 3 << "]: " << C[row][col + 3] << endl;
//}
//Matrix Matrix::multi_loopunrolling_packing_1x4simd256x1(Matrix &A, Matrix &B) {
//	if (A.cols() != B.rows()) {
//		return Matrix(0, 0);
//	}
//	Matrix C(A.rows(), B.cols(), 0);
//	//Matrix tr(A.cols(), 4, 0);
//	Matrix tr(1, 4 * A.cols(), 0);
//	int i(0), j(0), k;
//	int Acols= A.cols(), Crows= C.rows(), Ccols = C.cols();
//	do {
//		j = 0, k = 0;
//		do {
//			tr(0, j + 0) = B(k, i);//packing���̣��������ݴ���������ռ�
//			tr(0, j + 1) = B(k, i + 1);
//			tr(0, j + 2) = B(k, i + 2);
//			tr(0, j + 3) = B(k, i + 3);
//			j += 4;
//			k++;
//		} while (k<Acols);
//		//} while (k<A.cols());
//		//tr.print();
//		j = 0;
//		do {
//			loopunrolling_packing_1x4simd256x1_kernel(A, tr, C, j, i);
//			j += 1;
//		} while (j<Crows);
//		//} while (j<C.rows());
//		i += 4;
//	} while (i<Ccols);
//	//} while (i<C.cols());
//
//	return C;
//}
//
//void Matrix::loopunrolling_packing_1x8simd256x2_kernel(Matrix &A0, Matrix &B0, Matrix &C0, int row, int col) {
//	double** A(A0.ptr());
//	double** B(B0.ptr());
//	double** C(C0.ptr());
//	__m256d t0, t1, a0, b0, b1;
//	t0 = t1 = _mm256_set1_pd(0);
//	double *pa0(A[row]);
//	double *pb0(B[0]), *pb1(B[1]);
//	double *end = pa0 + A0.cols();
//	do {
//		b0 = _mm256_loadu_pd(pb0);
//		b1 = _mm256_loadu_pd(pb1);
//		a0 = _mm256_set1_pd(*(pa0++));
//		t0 = _mm256_add_pd(t0, _mm256_mul_pd(a0, b0));
//		t1 = _mm256_add_pd(t1, _mm256_mul_pd(a0, b1));
//		pb0 += 4;
//		pb1 += 4;
//	} while (pa0 != end);
//	_mm256_storeu_pd(&C[row][col], t0);
//	_mm256_storeu_pd(&C[row][col + 4], t1);
//	//cout << "C[" << row << "][" << col << "]: " << C[row][col] << endl;
//	//cout << "C[" << row << "][" << col + 1 << "]: " << C[row][col + 1] << endl;
//	//cout << "C[" << row << "][" << col + 2 << "]: " << C[row][col + 2] << endl;
//	//cout << "C[" << row << "][" << col + 3 << "]: " << C[row][col + 3] << endl;
//}
//Matrix Matrix::multi_loopunrolling_packing_1x8simd256x2(Matrix &A, Matrix &B) {
//	if (A.cols() != B.rows()) {
//		return Matrix(0, 0);
//	}
//	Matrix C(A.rows(), B.cols(), 0);
//	Matrix tr(2, 4 * A.cols(), 0);
//
//	size_t i(0), j(0), k;
//	int Acols = A.cols(), Crows = C.rows(), Ccols = C.cols();
//	do {
//		j = 0; k = 0;
//		do {
//			tr(0, k) = B(j, i);
//			tr(1, k++) = B(j, i + 4);
//
//			tr(0, k) = B(j, i + 1);
//			tr(1, k++) = B(j, i + 5);
//
//			tr(0, k) = B(j, i + 2);
//			tr(1, k++) = B(j, i + 6);
//
//			tr(0, k) = B(j, i + 3);
//			tr(1, k++) = B(j, i + 7);
//			j++;
//		//} while (j<A.cols());
//		} while (j<Acols);
//		j = 0;
//		do {
//			loopunrolling_packing_1x8simd256x2_kernel(A, tr, C, j, i);
//			j += 1;
//		//} while (j<C.rows());
//		} while (j<Crows);
//		i += 8;
//	//} while (i<C.cols());
//	} while (i<Ccols);
//
//	return C;
//}
//
//
//Matrix Matrix::multi_loopunrolling_4x4register(Matrix &A, Matrix &B) {
//	if (A.cols() != B.rows()) {
//		return Matrix(0, 0);
//	}
//	Matrix C(A.rows(), B.cols(), 0);
//	for (int m = 0; m < C.rows(); m += 4) {
//		for (int n = 0; n <C.cols(); n += 4) {
//			double temp_m0n0(0);
//			double temp_m0n1(0);
//			double temp_m0n2(0);
//			double temp_m0n3(0);
//
//			double temp_m1n0(0);
//			double temp_m1n1(0);
//			double temp_m1n2(0);
//			double temp_m1n3(0);
//
//			double temp_m2n0(0);
//			double temp_m2n1(0);
//			double temp_m2n2(0);
//			double temp_m2n3(0);
//
//			double temp_m3n0(0);
//			double temp_m3n1(0);
//			double temp_m3n2(0);
//			double temp_m3n3(0);
//
//			for (int k = 0; k < A.rows(); k++) {
//				double temp_m0 = A(m + 0, k);
//				double temp_m1 = A(m + 1, k);
//				double temp_m2 = A(m + 2, k);
//				double temp_m3 = A(m + 3, k);
//
//				double temp_n0 = B(k, n + 0);
//				double temp_n1 = B(k, n + 1);
//				double temp_n2 = B(k, n + 2);
//				double temp_n3 = B(k, n + 3);
//
//				temp_m0n0 += temp_m0 * temp_n0;
//				temp_m0n1 += temp_m0 * temp_n1;
//				temp_m0n2 += temp_m0 * temp_n2;
//				temp_m0n3 += temp_m0 * temp_n3;
//
//				temp_m1n0 += temp_m1 * temp_n0;
//				temp_m1n1 += temp_m1 * temp_n1;
//				temp_m1n2 += temp_m1 * temp_n2;
//				temp_m1n3 += temp_m1 * temp_n3;
//
//				temp_m2n0 += temp_m2 * temp_n0;
//				temp_m2n1 += temp_m2 * temp_n1;
//				temp_m2n2 += temp_m2 * temp_n2;
//				temp_m2n3 += temp_m2 * temp_n3;
//
//				temp_m3n0 += temp_m3 * temp_n0;
//				temp_m3n1 += temp_m3 * temp_n1;
//				temp_m3n2 += temp_m3 * temp_n2;
//				temp_m3n3 += temp_m3 * temp_n3;
//			}
//			C(m + 0, n + 0) = temp_m0n0;
//			C(m + 0, n + 1) = temp_m0n1;
//			C(m + 0, n + 2) = temp_m0n2;
//			C(m + 0, n + 3) = temp_m0n3;
//
//			C(m + 1, n + 0) = temp_m1n0;
//			C(m + 1, n + 1) = temp_m1n1;
//			C(m + 1, n + 2) = temp_m1n2;
//			C(m + 1, n + 3) = temp_m1n3;
//
//			C(m + 2, n + 0) = temp_m2n0;
//			C(m + 2, n + 1) = temp_m2n1;
//			C(m + 2, n + 2) = temp_m2n2;
//			C(m + 2, n + 3) = temp_m2n3;
//
//			C(m + 3, n + 0) = temp_m3n0;
//			C(m + 3, n + 1) = temp_m3n1;
//			C(m + 3, n + 2) = temp_m3n2;
//			C(m + 3, n + 3) = temp_m3n3;
//		}
//	}
//	return C;
//}
//
//void Matrix::loopunrolling_packing_4x8simd256x2_kernel(Matrix &A0, Matrix &B0, Matrix &C0, int row, int col) {
//	double** A(A0.ptr());
//	double** B(B0.ptr());
//	double** C(C0.ptr());
//	__m256d t0_0, t0_1, t0_2, t0_3, t1_0, t1_1, t1_2, t1_3, a0, a1, a2, a3, b0, b1;
//	t0_0 = t0_1 = t0_2 = t0_3 = t1_0 = t1_1 = t1_2 = t1_3 = _mm256_set1_pd(0);
//	double *pa0(A[row]), *pa1(A[row + 1]), *pa2(A[row + 2]), *pa3(A[row + 3]);
//	double *pb0(B[0]), *pb1(B[1]);
//	double *end = pa0 + A0.cols();
//	do {
//		b0 = _mm256_loadu_pd(pb0);
//		b1 = _mm256_loadu_pd(pb1);
//		a0 = _mm256_set1_pd(*(pa0++));
//		a1 = _mm256_set1_pd(*(pa1++));
//		a2 = _mm256_set1_pd(*(pa2++));
//		a3 = _mm256_set1_pd(*(pa3++));
//
//		t0_0 = _mm256_add_pd(t0_0, _mm256_mul_pd(a0, b0));
//		t0_1 = _mm256_add_pd(t0_1, _mm256_mul_pd(a1, b0));
//		t0_2 = _mm256_add_pd(t0_2, _mm256_mul_pd(a2, b0));
//		t0_3 = _mm256_add_pd(t0_3, _mm256_mul_pd(a3, b0));
//
//		t1_0 = _mm256_add_pd(t1_0, _mm256_mul_pd(a0, b1));
//		t1_1 = _mm256_add_pd(t1_1, _mm256_mul_pd(a1, b1));
//		t1_2 = _mm256_add_pd(t1_2, _mm256_mul_pd(a2, b1));
//		t1_3 = _mm256_add_pd(t1_3, _mm256_mul_pd(a3, b1));
//
//		pb0 += 4;
//		pb1 += 4;
//	} while (pa0 != end);
//	_mm256_storeu_pd(&C[row][col], t0_0);
//	_mm256_storeu_pd(&C[row + 1][col], t0_1);
//	_mm256_storeu_pd(&C[row + 2][col], t0_2);
//	_mm256_storeu_pd(&C[row + 3][col], t0_3);
//	_mm256_storeu_pd(&C[row][col + 4], t1_0);
//	_mm256_storeu_pd(&C[row + 1][col + 4], t1_1);
//	_mm256_storeu_pd(&C[row + 2][col + 4], t1_2);
//	_mm256_storeu_pd(&C[row + 3][col + 4], t1_3);
//	//cout << "C[" << row << "][" << col << "]: " << C[row][col] << endl;
//	//cout << "C[" << row << "][" << col + 1 << "]: " << C[row][col + 1] << endl;
//	//cout << "C[" << row << "][" << col + 2 << "]: " << C[row][col + 2] << endl;
//	//cout << "C[" << row << "][" << col + 3 << "]: " << C[row][col + 3] << endl;
//}
//Matrix Matrix::multi_loopunrolling_packing_4x8simd256x2(Matrix &A, Matrix &B) {
//	if (A.cols() != B.rows()) {
//		return Matrix(0, 0);
//	}
//	Matrix C(A.rows(), B.cols(), 0);
//	Matrix tr(2, 4 * A.cols(), 0);
//
//	size_t i(0), j(0), k;
//	do {
//		j = 0; k = 0;
//		do {
//			tr(0, k) = B(j, i);
//			tr(1, k++) = B(j, i + 4);
//
//			tr(0, k) = B(j, i + 1);
//			tr(1, k++) = B(j, i + 5);
//
//			tr(0, k) = B(j, i + 2);
//			tr(1, k++) = B(j, i + 6);
//
//			tr(0, k) = B(j, i + 3);
//			tr(1, k++) = B(j, i + 7);
//			j++;
//		} while (j<A.cols());
//		j = 0;
//		do {
//			loopunrolling_packing_4x8simd256x2_kernel(A, tr, C, j, i);
//			j += 4;
//		} while (j<C.rows());
//		i += 8;
//	} while (i<C.cols());
//	return C;
//}
//
//Matrix Matrix::conv_nature(Matrix & A, Matrix & B)
//{
//	if (B.cols() % 2 == 0 || B.rows() % 2 == 0||B.cols() <= 0 || B.rows() <= 0 || A.cols() <= 0 || A.rows() <= 0) {
//		return Matrix(0, 0);
//	}
//	int height = A.rows();
//	int width = A.cols();
//	int kernel_h = B.rows();
//	int kernel_w = B.cols();
//	int pad_h = kernel_h / 2;
//	int pad_w = kernel_w / 2;
//	//if (pad_h >= height || pad_w >= width) { return Matrix(0, 0);}
//	Matrix C_tmp= padding(A, pad_h, pad_w); //C_tmp.print();
//
//
//	Matrix C(height, width, 0);
//
//	//��˹��ת������
//	Matrix kernel=B.flatten();
//	//kernel.print();
//	Matrix tmp(1, kernel_h*kernel_w);
//	for (size_t i = 0; i < height; i++) {
//		for (size_t j = 0; j < width; j++) {
//			for (size_t k = 0; k < kernel_h; k++) {
//				for (size_t z = 0; z < kernel_w; z++) {
//					tmp(0, k*kernel_w + z) = C_tmp(i + k, j + z);
//				}
//			}
//			double sum = 0;
//			for (size_t g = 0; g < kernel_h*kernel_w; g++) {
//				tmp(0, g) *= kernel(0, g);
//				sum += tmp(0, g);
//			}
//			C(i, j) = sum;
//		}
//	}
//
//	
//
//
//	return C;
//
//
//
//
//	
//}
//
//Matrix Matrix::padding(Matrix & A, int pad_h, int pad_w) {
//	int height = A.rows();
//	int width = A.cols();
//	Matrix C_tmp(2 * pad_h + height, 2 * pad_w + width, 0);
//
//	double** ptr_C_tmp = C_tmp.ptr();
//	double** ptr_A = A.ptr();
//	//���ͼ��gfedcb|abcdefgh|gfedcba �Ա߽�Ϊ�Գ��ᷴ�临�����أ��ο�opencv��cv::BorderTypes��BORDER_REFLECT_101
//	for (int i = pad_h; i < pad_h + height; i++) {
//		for (int j = pad_w; j < pad_w + width; j++) {
//			//C_tmp(i, j) = A(i - pad_h, j - pad_w);
//			*(*(ptr_C_tmp+i)+ j) = *(*(ptr_A+i - pad_h)+ j - pad_w);
//		}
//	}
//
//
//	//Matrix LU(pad_h, pad_w), RU(pad_h, pad_w), LD(pad_h, pad_w), RD(pad_h, pad_w);
//	//for (int i = pad_h + 1, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = pad_w + 1, z = 0; z < pad_w; j++, z++) {
//	//		LU(k, z) = C_tmp(i, j);
//	//	}
//	//}
//	//for (int i = pad_h + 1, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = width - 1, z = 0; z < pad_w; j++, z++) {
//	//		RU(k, z) = C_tmp(i, j);
//	//	}
//	//}
//	//for (int i = height - 1, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = pad_w + 1, z = 0; z < pad_w; j++, z++) {
//	//		LD(k, z) = C_tmp(i, j);
//	//	}
//	//}
//	//for (int i = width - 1, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = height - 1, z = 0; z < pad_w; j++, z++) {
//	//		RD(k, z) = C_tmp(i, j);
//	//	}
//	//}
//
//	////LU.print(); RU.print(); LD.print(); RD.print();
//	//Matrix LU0(pad_h, pad_w), RU0(pad_h, pad_w), LD0(pad_h, pad_w), RD0(pad_h, pad_w);
//	//for (int k = 0; k < pad_h; k++) {
//	//	for (int z = 0; z < pad_w; z++) {
//	//		LU0(pad_h - k - 1, pad_w - z - 1) = LU(k, z);
//	//		RU0(pad_h - k - 1, pad_w - z - 1) = RU(k, z);
//	//		LD0(pad_h - k - 1, pad_w - z - 1) = LD(k, z);
//	//		RD0(pad_h - k - 1, pad_w - z - 1) = RD(k, z);
//	//	}
//	//}
//
//	//for (int i = 0, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = 0, z = 0; z < pad_w; j++, z++) {
//	//		C_tmp(i, j) = LU0(k, z);
//	//	}
//	//}
//	//for (int i = 0, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = pad_w + width, z = 0; z < pad_w; j++, z++) {
//	//		C_tmp(i, j) = RU0(k, z);
//	//	}
//	//}
//	//for (int i = pad_h + height, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = 0, z = 0; z < pad_w; j++, z++) {
//	//		C_tmp(i, j) = LD0(k, z);
//	//	}
//	//}
//	//for (int i = pad_h + height, k = 0; k < pad_h; i++, k++) {
//	//	for (int j = pad_w + width, z = 0; z < pad_w; j++, z++) {
//	//		C_tmp(i, j) = RD0(k, z);
//	//	}
//	//}
//
//	//for (int i = 0; i < 2 * pad_h + height; i++) {
//	//	for (int j = 0; j < 2 * pad_w + width; j++) {
//	//		if (pad_h <= i&&i < pad_h + height) {
//	//			for (int index0 = pad_w + 1, index1 = pad_w + width, k = 0; k < pad_w; index0++, index1++, k++) {
//	//				C_tmp(i, index0 - 2 * (k + 1)) = C_tmp(i, index0);
//	//				C_tmp(i, index1) = C_tmp(i, index1 - 2 * (k + 1));
//	//			}
//	//		}
//	//		if (pad_w <= j&&j < pad_w + width) {
//	//			for (int index0 = pad_h + 1, index1 = pad_h + height, k = 0; k < pad_h; index0++, index1++, k++) {
//	//				C_tmp(index0 - 2 * (k + 1), j) = C_tmp(index0, j);
//	//				C_tmp(index1, j) = C_tmp(index1 - 2 * (k + 1), j);
//	//			}
//	//		}
//	//	}
//	//}
//
//
//	return C_tmp;
//}
