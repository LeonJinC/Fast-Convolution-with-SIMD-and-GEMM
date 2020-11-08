#include"MyMatrix.h"
using namespace std;

void* aligned_malloc(size_t size, int alignment)
{
	// 分配足够的内存, 这里的算法很经典, 早期的STL中使用的就是这个算法  

	// 首先是维护FreeBlock指针占用的内存大小  
	const int pointerSize = sizeof(void*);

	// alignment - 1 + pointerSize这个是FreeBlock内存对齐需要的内存大小  
	// 前面的例子sizeof(T) = 20, __alignof(T) = 16,  
	// g_MaxNumberOfObjectsInPool = 1000  
	// 那么调用本函数就是alignedMalloc(1000 * 20, 16)  
	// 那么alignment - 1 + pointSize = 19  
	const int requestedSize = size + alignment - 1 + pointerSize;

	// 分配的实际大小就是20000 + 19 = 20019  
	void* raw = malloc(requestedSize);

	// 这里实Pool真正为对象实例分配的内存地址  
	uintptr_t start = (uintptr_t)raw + pointerSize;
	// 向上舍入操作  
	// 解释一下, __ALIGN - 1指明的是实际内存对齐的粒度  
	// 例如__ALIGN = 8时, 我们只需要7就可以实际表示8个数(0~7)  
	// 那么~(__ALIGN - 1)就是进行舍入的粒度  
	// 我们将(bytes) + __ALIGN-1)就是先进行进位, 然后截断  
	// 这就保证了我是向上舍入的  
	// 例如byte = 100, __ALIGN = 8的情况  
	// ~(__ALIGN - 1) = (1 000)B  
	// ((bytes) + __ALIGN-1) = (1 101 011)B  
	// (((bytes) + __ALIGN-1) & ~(__ALIGN - 1)) = (1 101 000 )B = (104)D  
	// 104 / 8 = 13, 这就实现了向上舍入  
	// 对于byte刚好满足内存对齐的情况下, 结果保持byte大小不变  
	// 记得《Hacker's Delight》上面有相关的计算  
	// 这个表达式与下面给出的等价  
	// ((((bytes) + _ALIGN - 1) * _ALIGN) / _ALIGN)  
	// 但是SGI STL使用的方法效率非常高   
	void* aligned = (void*)((start + alignment - 1) & ~(alignment - 1));

	// 这里维护一个指向malloc()真正分配的内存  
	*(void**)((uintptr_t)aligned - pointerSize) = raw;

	// 返回实例对象真正的地址  
	return aligned;
}


// 这里是内部维护的内存情况  
//                   这里满足内存对齐要求  
//                             |  
// ----------------------------------------------------------------------  
// | 内存对齐填充 | 维护的指针 | 对象1 | 对象2 | 对象3 | ...... | 对象n |  
// ----------------------------------------------------------------------  
// ^                     | 指向malloc()分配的地址起点  
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
	// 又是一个经典算法, 参见<Hacker's Delight>  
	return ((uintptr_t)data & (alignment - 1)) == 0;
}

Matrix::Matrix(size_t r, size_t c) :_Row(r), _Column(c) {//构造r行、c列的矩阵
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
Matrix::Matrix(size_t r, size_t c, const double init) :_Row(r), _Column(c) {//构造r行、c列的矩阵并用init初始化
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
Matrix::Matrix(const Matrix& B) {//拷贝构造
						 //cout << "拷贝构造" << endl;
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
Matrix::~Matrix() {//析构函数
	if (!_row_Matrix) return;
	//double **p = _Matrix, **end = _Matrix + _Row;
	//do {
	//	delete[](*(p++));
	//} while (p != end);
	//delete[] _Matrix;
	//delete[] _row_Matrix;
	aligned_free(_row_Matrix);
	_Column = _Row = 0;
	//cout << "析构函数" << endl;
}

Matrix& Matrix::operator=(Matrix&& B) {//移动拷贝赋值
							   //cout << "移动赋值" << endl;
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
			std::cout << _row_Matrix[i*_Column +j] << "\t";
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
	
	for (size_t i = 0; i < Crows; i++) {
		for (size_t j = 0; j < Ccols; j++) {
			for (size_t k = 0; k < Acols; k++) {
				C[i*Ccols +j] += A[i*Acols + k] * B[k*Bcols + j];
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
	for (size_t i = 0; i < Crows; i++) {
		for (size_t k = 0; k < Acols; k++) {
			for (size_t j = 0; j < Ccols; j++) {
				C[i*Ccols + j] += A[i*Acols + k] * B[k*Bcols + j];
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
	for (size_t i = 0; i < Crows; i++) {
		for (size_t j = 0; j < Ccols; j++) {
			temp = 0;
			for (size_t k = 0; k < Acols; k++) {
				temp += A[i*Acols + k] * B[k*Bcols + j];
			}
			C[i*Ccols + j] = temp;
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
	for (size_t i = 0; i < Crows; i++) {
		for (size_t j = 0; j < Ccols; j += 4) {
			t0 = 0; t1 = 0; t2 = 0; t3 = 0;
			for (size_t k = 0; k < Acols; k++) {
				temp = A[i*Acols + k];
				t0 += temp * B[k*Bcols + (j + 0)];
				t1 += temp * B[k*Bcols + (j + 1)];
				t2 += temp * B[k*Bcols + (j + 2)];
				t3 += temp * B[k*Bcols + (j + 3)];
			}
			C[i*Ccols + (j + 0)] = t0;
			C[i*Ccols + (j + 1)] = t1;
			C[i*Ccols + (j + 2)] = t2;
			C[i*Ccols + (j + 3)] = t3;
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
	
	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	__m256d a0, b0,c0;
	for (size_t i = 0; i < Crows; i++) {
		for (size_t j = 0; j < Ccols; j += 4) {
			c0 = _mm256_load_pd(c+i*Ccols+j);
			for (size_t k = 0; k < Acols; k++){
				a0 = _mm256_broadcast_sd(a + i*Acols + k);
				b0 = _mm256_load_pd(b + j + k*Bcols);
				c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
			}
			_mm256_store_pd(c + i*Ccols + j, c0);
		}
	}
	return C;
}

Matrix Matrix::multi_avx_blk(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	__m256d a0, b0, c0;
	size_t unroll = 1;
	size_t blocksize = unroll * 8;

	
	for (size_t si = 0; si < Crows; si += blocksize) {
		for (size_t sj = 0; sj < Ccols; sj += blocksize) {
			for (int sk = 0; sk < Acols; sk += blocksize) {
				for (int i = si; i < si + blocksize; i++) {
					for (int j = sj; j < sj + blocksize; j += unroll * 4) {
					
						c0 = _mm256_load_pd(c + i*Ccols + j);
						for (int k = sk; k < sk + blocksize; k++) {
							a0 = _mm256_broadcast_sd(a + i*Acols + k);

							b0 = _mm256_load_pd(b + j + k*Bcols);
							c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
						}
						_mm256_store_pd(c + i*Ccols + j, c0);
					}
					
				}

			}
		}
	}
	return C;
}


Matrix Matrix::multi_avx_unrollx2(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();

	__m256d a0, cm[2];

	for (int i = 0; i < Crows; i++) {
		for (int j = 0; j < Ccols; j += 8) {//4*4
		
			cm[0] = _mm256_load_pd(c + i*Ccols + j);
			cm[1] = _mm256_load_pd(c + i*Ccols + 4 + j);
			for (int k = 0; k < Acols; k++) {
				a0 = _mm256_broadcast_sd(a + i*Acols + k);

				cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols)));
				cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 4)));
			}

			_mm256_store_pd(c + i*Ccols + j, cm[0]);
			_mm256_store_pd(c + i*Ccols + 4 + j, cm[1]);
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

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	__m256d a0, cm[UNROLL];
	//#pragma omp parallel for
	for (int i = 0; i < Crows; i++) {
		for (int j = 0; j < Ccols; j += UNROLL*4) {//4*4
											 //for (int x = 0; x < UNROLL; x++) {
											 //	cm[x] = _mm256_load_pd(c + i*Ccols+x*4);
											 //}
			cm[0] = _mm256_load_pd(c + i*Ccols + j);
			cm[1] = _mm256_load_pd(c + i*Ccols + 4 + j);
			cm[2] = _mm256_load_pd(c + i*Ccols + 8 + j);
			cm[3] = _mm256_load_pd(c + i*Ccols + 12 + j);
			for (int k = 0; k < Acols; k++) {
				a0 = _mm256_broadcast_sd(a + i*Acols + k);
				//for (int x = 0; x < UNROLL; x++) {
				//	cm[x] = _mm256_add_pd(cm[x],_mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols+x*4)));
				//}
				cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols)));
				cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 4)));
				cm[2] = _mm256_add_pd(cm[2], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 8)));
				cm[3] = _mm256_add_pd(cm[3], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 12)));
			}
			//for (int x = 0; x < UNROLL; x++) {
			//	_mm256_store_pd(c + i*Ccols + x * 4, cm[x]);
			//}
			_mm256_store_pd(c + i*Ccols + j, cm[0]);
			_mm256_store_pd(c + i*Ccols + 4 + j, cm[1]);
			_mm256_store_pd(c + i*Ccols + 8 + j, cm[2]);
			_mm256_store_pd(c + i*Ccols + 12 + j, cm[3]);
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

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();

	for (size_t si = 0; si < Crows; si += BLOCKSIZE) {
			for (size_t sk = 0; sk < Acols; sk += BLOCKSIZE) {
				for (size_t sj = 0; sj < Ccols; sj += BLOCKSIZE) {

				for (int i = si; i < si + BLOCKSIZE; i++) {
					for (int j = sj; j < sj + BLOCKSIZE; j += UNROLL * 4) {
						__m256d cm[UNROLL];
						for (int x = 0; x < UNROLL; x++) {
							cm[x] = _mm256_load_pd(c + i*Ccols + x * 4 + j);
						}
						for (int k = sk; k < sk + BLOCKSIZE; k++) {
							__m256d a0 = _mm256_broadcast_sd(a + i*Acols + k);
							for (int x = 0; x < UNROLL; x++) {
								cm[x] = _mm256_add_pd(cm[x], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + x * 4)));
							}
						}
						for (int x = 0; x < UNROLL; x++) {
							_mm256_store_pd(c + i*Ccols + x * 4 + j, cm[x]);
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
	int Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	//int unroll = 4;
	//int blocksize = unroll * 8;
#pragma omp parallel for num_threads(8)
	for (int si = 0; si < Crows; si += BLOCKSIZE) {
		for (int sk = 0; sk < Acols; sk += BLOCKSIZE) {
			for (int sj = 0; sj < Ccols; sj += BLOCKSIZE) {
				for (int i = si; i < si + BLOCKSIZE; i++) {
					for (int j = sj; j < sj + BLOCKSIZE; j += UNROLL * 4) {
						__m256d cm[UNROLL];
						for (int x = 0; x < UNROLL; x++) {
							cm[x] = _mm256_load_pd(c + i*Ccols + x * 4 + j);
						}
						for (int k = sk; k < sk + BLOCKSIZE; k++) {
							__m256d a0 = _mm256_broadcast_sd(a + i*Acols + k);
							for (int x = 0; x < UNROLL; x++) {
								cm[x] = _mm256_add_pd(cm[x], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + x * 4)));
							}
						}
						for (int x = 0; x < UNROLL; x++) {
							_mm256_store_pd(c + i*Ccols + x * 4 + j, cm[x]);
						}
					}
				}
			}
		}
	}

	return C;
}

Matrix Matrix::multi_avx_unrollx8_ijk(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	__m256d a0, cm[8];
	//#pragma omp parallel for
	for (int j = 0; j < Ccols; j += 32) {//4*4
		for (int i = 0; i < Crows; i++) {
		
											 //for (int x = 0; x < 8; x++) {
											 //cm[x] = _mm256_load_pd(c + i*Ccols+x*4);
											 //}
			cm[0] = _mm256_load_pd(c + i*Ccols + j);
			cm[1] = _mm256_load_pd(c + i*Ccols + 4 + j);
			cm[2] = _mm256_load_pd(c + i*Ccols + 8 + j);
			cm[3] = _mm256_load_pd(c + i*Ccols + 12 + j);
			cm[4] = _mm256_load_pd(c + i*Ccols + 16 + j);
			cm[5] = _mm256_load_pd(c + i*Ccols + 20 + j);
			cm[6] = _mm256_load_pd(c + i*Ccols + 24 + j);
			cm[7] = _mm256_load_pd(c + i*Ccols + 28 + j);
			for (int k = 0; k < Acols; k++) {
				a0 = _mm256_broadcast_sd(a + i*Acols + k);
				//for (int x = 0; x < 8; x++) {
				//	cm[x] = _mm256_add_pd(cm[x],_mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols+x*4)));
				//}
				cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols)));
				cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 4)));
				cm[2] = _mm256_add_pd(cm[2], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 8)));
				cm[3] = _mm256_add_pd(cm[3], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 12)));
				cm[4] = _mm256_add_pd(cm[4], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 16)));
				cm[5] = _mm256_add_pd(cm[5], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 20)));
				cm[6] = _mm256_add_pd(cm[6], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 24)));
				cm[7] = _mm256_add_pd(cm[7], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 28)));
			}
			//for (int x = 0; x < 8; x++) {
			//	_mm256_store_pd(c + i*Ccols + x * 4, cm[x]);
			//}
			_mm256_store_pd(c + i*Ccols + j, cm[0]);
			_mm256_store_pd(c + i*Ccols + 4 + j, cm[1]);
			_mm256_store_pd(c + i*Ccols + 8 + j, cm[2]);
			_mm256_store_pd(c + i*Ccols + 12 + j, cm[3]);
			_mm256_store_pd(c + i*Ccols + 16 + j, cm[4]);
			_mm256_store_pd(c + i*Ccols + 20 + j, cm[5]);
			_mm256_store_pd(c + i*Ccols + 24 + j, cm[6]);
			_mm256_store_pd(c + i*Ccols + 28 + j, cm[7]);
		}
	}
	return C;
}

Matrix Matrix::multi_avx_unrollx8_jik(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	__m256d a0, cm[8];
	//#pragma omp parallel for
	for (int j = 0; j < Ccols; j += 32) {//4*4
		for (int i = 0; i < Crows; i++) {
			//for (int x = 0; x < 8; x++) {
			//cm[x] = _mm256_load_pd(c + i*Ccols+x*4);
			//}
			cm[0] = _mm256_load_pd(c + i*Ccols + j);
			cm[1] = _mm256_load_pd(c + i*Ccols + 4 + j);
			cm[2] = _mm256_load_pd(c + i*Ccols + 8 + j);
			cm[3] = _mm256_load_pd(c + i*Ccols + 12 + j);
			cm[4] = _mm256_load_pd(c + i*Ccols + 16 + j);
			cm[5] = _mm256_load_pd(c + i*Ccols + 20 + j);
			cm[6] = _mm256_load_pd(c + i*Ccols + 24 + j);
			cm[7] = _mm256_load_pd(c + i*Ccols + 28 + j);
			for (int k = 0; k < Acols; k++) {
				a0 = _mm256_broadcast_sd(a + i*Acols + k);
				//for (int x = 0; x < 8; x++) {
				//	cm[x] = _mm256_add_pd(cm[x],_mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols+x*4)));
				//}
				cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols)));
				cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 4)));
				cm[2] = _mm256_add_pd(cm[2], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 8)));
				cm[3] = _mm256_add_pd(cm[3], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 12)));
				cm[4] = _mm256_add_pd(cm[4], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 16)));
				cm[5] = _mm256_add_pd(cm[5], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 20)));
				cm[6] = _mm256_add_pd(cm[6], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 24)));
				cm[7] = _mm256_add_pd(cm[7], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 28)));
			}
			//for (int x = 0; x < 8; x++) {
			//	_mm256_store_pd(c + i*Ccols + x * 4, cm[x]);
			//}
			_mm256_store_pd(c + i*Ccols + j, cm[0]);
			_mm256_store_pd(c + i*Ccols + 4 + j, cm[1]);
			_mm256_store_pd(c + i*Ccols + 8 + j, cm[2]);
			_mm256_store_pd(c + i*Ccols + 12 + j, cm[3]);
			_mm256_store_pd(c + i*Ccols + 16 + j, cm[4]);
			_mm256_store_pd(c + i*Ccols + 20 + j, cm[5]);
			_mm256_store_pd(c + i*Ccols + 24 + j, cm[6]);
			_mm256_store_pd(c + i*Ccols + 28 + j, cm[7]);
		}
	}
	return C;
}

Matrix Matrix::multi_avx_unrollx8_omp(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	//__m256d a0, cm[8];
#pragma omp parallel for num_threads(8)
	for (int j = 0; j < Ccols; j += 32) {//4*4
		for (int i = 0; i < Crows; i++) {

			__m256d cm[8];
			for (int x = 0; x < 8; x++) {
				cm[x] = _mm256_load_pd(c + i*Ccols + x * 4 + j);
			}
			//cm[0] = _mm256_load_pd(c + i*Ccols);
			//cm[1] = _mm256_load_pd(c + i*Ccols + 4);
			//cm[2] = _mm256_load_pd(c + i*Ccols + 8);
			//cm[3] = _mm256_load_pd(c + i*Ccols + 12);
			//cm[4] = _mm256_load_pd(c + i*Ccols + 16);
			//cm[5] = _mm256_load_pd(c + i*Ccols + 20);
			//cm[6] = _mm256_load_pd(c + i*Ccols + 24);
			//cm[7] = _mm256_load_pd(c + i*Ccols + 28);
			for (int k = 0; k < Acols; k++) {
				__m256d a0 = _mm256_broadcast_sd(a + i*Acols + k);
				for (int x = 0; x < 8; x++) {
					cm[x] = _mm256_add_pd(cm[x], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + x * 4)));
				}
				//cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols)));
				//cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 4)));
				//cm[2] = _mm256_add_pd(cm[2], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 8)));
				//cm[3] = _mm256_add_pd(cm[3], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 12)));
				//cm[4] = _mm256_add_pd(cm[4], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 16)));
				//cm[5] = _mm256_add_pd(cm[5], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 20)));
				//cm[6] = _mm256_add_pd(cm[6], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 24)));
				//cm[7] = _mm256_add_pd(cm[7], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 28)));
			}
			for (int x = 0; x < 8; x++) {
				_mm256_store_pd(c + i*Ccols + x * 4 + j, cm[x]);
			}
			//_mm256_store_pd(c + i*Ccols, cm[0]);
			//_mm256_store_pd(c + i*Ccols + 4, cm[1]);
			//_mm256_store_pd(c + i*Ccols + 8, cm[2]);
			//_mm256_store_pd(c + i*Ccols + 12, cm[3]);
			//_mm256_store_pd(c + i*Ccols + 16, cm[4]);
			//_mm256_store_pd(c + i*Ccols + 20, cm[5]);
			//_mm256_store_pd(c + i*Ccols + 24, cm[6]);
			//_mm256_store_pd(c + i*Ccols + 28, cm[7]);
		}
	}
	return C;
}

Matrix Matrix::multi_avx_unrollx16_jik_omp(Matrix & A, Matrix & B)
{
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);
	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();

	double  *a = A.ptr();
	double  *b = B.ptr();
	double  *c = C.ptr();
	//__m256d a0, cm[8];
	int unroll = 16;
	//#pragma omp parallel for num_threads(8)
	for (int j = 0; j < Ccols; j += 4 * unroll) {//4*4
		for (int i = 0; i < Crows; i++) {

			__m256d cm[16];
			for (int x = 0; x < unroll; x++) {
				cm[x] = _mm256_load_pd(c + i*Ccols + x * 4 + j);
			}
			//cm[0] = _mm256_load_pd(c + i*Ccols);
			//cm[1] = _mm256_load_pd(c + i*Ccols + 4);
			//cm[2] = _mm256_load_pd(c + i*Ccols + 8);
			//cm[3] = _mm256_load_pd(c + i*Ccols + 12);
			//cm[4] = _mm256_load_pd(c + i*Ccols + 16);
			//cm[5] = _mm256_load_pd(c + i*Ccols + 20);
			//cm[6] = _mm256_load_pd(c + i*Ccols + 24);
			//cm[7] = _mm256_load_pd(c + i*Ccols + 28);
			for (int k = 0; k < Acols; k++) {
				__m256d a0 = _mm256_broadcast_sd(a + i*Acols + k);
				for (int x = 0; x < unroll; x++) {
					cm[x] = _mm256_add_pd(cm[x], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + x * 4)));
				}
				//cm[0] = _mm256_add_pd(cm[0], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols)));
				//cm[1] = _mm256_add_pd(cm[1], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 4)));
				//cm[2] = _mm256_add_pd(cm[2], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 8)));
				//cm[3] = _mm256_add_pd(cm[3], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 12)));
				//cm[4] = _mm256_add_pd(cm[4], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 16)));
				//cm[5] = _mm256_add_pd(cm[5], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 20)));
				//cm[6] = _mm256_add_pd(cm[6], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 24)));
				//cm[7] = _mm256_add_pd(cm[7], _mm256_mul_pd(a0, _mm256_load_pd(b + j + k*Bcols + 28)));
			}
			for (int x = 0; x < unroll; x++) {
				_mm256_store_pd(c + i*Ccols + x * 4 + j, cm[x]);
			}
			//_mm256_store_pd(c + i*Ccols, cm[0]);
			//_mm256_store_pd(c + i*Ccols + 4, cm[1]);
			//_mm256_store_pd(c + i*Ccols + 8, cm[2]);
			//_mm256_store_pd(c + i*Ccols + 12, cm[3]);
			//_mm256_store_pd(c + i*Ccols + 16, cm[4]);
			//_mm256_store_pd(c + i*Ccols + 20, cm[5]);
			//_mm256_store_pd(c + i*Ccols + 24, cm[6]);
			//_mm256_store_pd(c + i*Ccols + 28, cm[7]);
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
//			tr(0, j) = B(j, i);//packing过程，把行数据打包到连续空间
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
//			tr(0, k) = B(j, i);//packing过程，把行数据打包到连续空间
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
//			tr(0, j + 0) = B(k, i);//packing过程，把行数据打包到连续空间
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
//	//高斯核转行向量
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
//	//填充图像，gfedcb|abcdefgh|gfedcba 以边界为对称轴反射复制像素，参考opencv的cv::BorderTypes的BORDER_REFLECT_101
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
