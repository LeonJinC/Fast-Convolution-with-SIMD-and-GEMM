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
	//_Matrix = new double*[_Row];
	//double **p = _Matrix, **end = _Matrix + _Row;
	//do {
	//	*(p++) = new double[_Column];
	//} while (p != end);
	_row_Matrix = (double*)aligned_malloc(sizeof(double)*_Row*_Column, 32);
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
	//_Matrix = new double*[_Row];
	//double **pr = _Matrix, **endr = _Matrix + _Row, *p, *end;
	//do {
	//	p = *(pr++) = new double[_Column];
	//	end = p + _Column;
	//	do {
	//		*(p++) = init;
	//	} while (p != end);
	//} while (pr != endr);
	_row_Matrix = (double*)aligned_malloc(sizeof(double)*_Row*_Column, 32);
	//std::cout << "address of _row_Matrix: " << _row_Matrix << std::endl;
	//if (isAligned(_row_Matrix, 32)) {
	//	cout << "isAligned" << endl;
	//}
	//else {
	//	cout << "not Aligned" << endl;
	//}
	memset(_row_Matrix, init, _Row*_Column * sizeof(double));
}
Matrix::Matrix(const Matrix& B) {//��������
						 //cout << "��������" << endl;
	//_Row = B._Row;
	//_Column = B._Column;
	//_Matrix = new double*[_Row];
	//double **pbr = B._Matrix, **endbr = B._Matrix + _Row, *pb, *endb,
	//	**par = _Matrix, **endar = _Matrix + _Row, *pa, *enda;
	//do {
	//	pa = *(par++) = new double[_Column];
	//	enda = pa + _Column;
	//	pb = *(pbr++);
	//	endb = pb + _Column;
	//	do {
	//		*(pa++) = *(pb++);
	//	} while (pa != enda);
	//} while (par != endar);

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
	//if (!_Matrix) return;
	//double **p = _Matrix, **end = _Matrix + _Row;
	//do {
	//	delete[](*(p++));
	//} while (p != end);
	//delete[] _Matrix;
	//_Column = _Row = 0;
	////cout << "��������" << endl;

	if (!_row_Matrix) return;
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

void Matrix::flatten() {
	_Column = _Row*_Column;
	_Row = 1;
	//Matrix flat(1, _Row*_Column);
	//for (int i = 0; i < _Row; i++) {
	//	for (int j = 0; j < _Column; j++) {
	//		flat(0, i * _Row + j) = _Matrix[i][j];
	//	}
	//}
	//return flat;
}

void Matrix::reshape(size_t rows, size_t cols)
{
	if (rows*cols != _Row*_Column) {
		std::cout << "rows*cols != _Row*_Column" << std::endl;
		return;
	}
	_Row = rows;
	_Column = cols;
}

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
			std::cout << _row_Matrix[i*_Column + j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

Matrix Matrix::multi_ijk(Matrix &A, Matrix &B) {
	if (A.cols() != B.rows()) {
		return Matrix(0, 0);
	}
	Matrix C(A.rows(), B.cols(), 0);

	size_t Crows = C.rows(), Ccols = C.cols(), Bcols = B.cols(), Brows = B.rows(), Acols = A.cols(), Arows = A.rows();
	for (size_t i = 0; i < Crows; i++) {
		for (size_t j = 0; j < Ccols; j++) {
			for (size_t k = 0; k < Acols; k++) {
				C[i*Ccols + j] += A[i*Acols + k] * B[k*Bcols + j];
			}
		}
	}
	//C.print();
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
	__m256d a0, b0, c0;
	for (size_t i = 0; i < Crows; i++) {
		for (size_t j = 0; j < Ccols; j += 4) {
			c0 = _mm256_load_pd(c + i*Ccols + j);
			for (size_t k = 0; k < Acols; k++) {
				a0 = _mm256_broadcast_sd(a + i*Acols + k);
				b0 = _mm256_load_pd(b + j + k*Bcols);
				c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
			}
			_mm256_store_pd(c + i*Ccols + j, c0);
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
	__m256d a0, cm[4];
	//#pragma omp parallel for
	for (int i = 0; i < Crows; i++) {
		for (int j = 0; j < Ccols; j += 16) {//4*4
											 //for (int x = 0; x < UNROLL; x++) {
											 //	cm[x] = _mm256_load_pd(c + i*Ccols+x*4);
											 //}
			cm[0] = _mm256_load_pd(c + i*Ccols+j);
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
	for (int i = 0; i < Crows; i++) {
		for (int j = 0; j < Ccols; j += 32) {//4*4
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
	for (int j = 0; j < Ccols; j += 4* unroll) {//4*4
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


Matrix Matrix::conv_nature(Matrix & A, Matrix & B, Matrix &C_tmp)
{
	if (B.cols() % 2 == 0 || B.rows() % 2 == 0||B.cols() <= 0 || B.rows() <= 0 || A.cols() <= 0 || A.rows() <= 0) {
		return Matrix(0, 0);
	}
	int height = A.rows();
	int width = A.cols();
	int kernel_h = B.rows();
	int kernel_w = B.cols();
	int pad_h = kernel_h / 2;
	int pad_w = kernel_w / 2;
	//if (pad_h >= height || pad_w >= width) { return Matrix(0, 0);}

	//Matrix C_tmp(2 * pad_h + height, 2 * pad_w + width, 0);
	Matrix::padding(A, pad_h, pad_w, C_tmp); //C_tmp.print();


	Matrix C(height, width, 0);

	//��˹��ת������
	B.flatten();
	//Matrix &kernel = B;
	//kernel.print();
	Matrix tmp(1, kernel_h*kernel_w);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			for (size_t k = 0; k < kernel_h; k++) {
				for (size_t z = 0; z < kernel_w; z++) {
					tmp(0, k*kernel_w + z) = C_tmp(i + k, j + z);
				}
			}
			double sum = 0;
			for (size_t g = 0; g < kernel_h*kernel_w; g++) {
				tmp(0, g) *= B(0, g);
				sum += tmp(0, g);
			}
			C(i, j) = sum;
		}
	}


	return C;




	
}

void Matrix::padding(Matrix & A, int pad_h, int pad_w, Matrix &C_tmp) {
	int height = A.rows();
	int width = A.cols();
	//Matrix C_tmp(2 * pad_h + height, 2 * pad_w + width, 0);

	//double* ptr_C_tmp = C_tmp.ptr();
	//double* ptr_A = A.ptr();
	//���ͼ��gfedcb|abcdefgh|gfedcba �Ա߽�Ϊ�Գ��ᷴ�临�����أ��ο�opencv��cv::BorderTypes��BORDER_REFLECT_101
	for (int i = pad_h; i < pad_h + height; i++) {
		for (int j = pad_w; j < pad_w + width; j++) {
			C_tmp(i, j) = A(i - pad_h, j - pad_w);
			//*(*(ptr_C_tmp+i)+ j) = *(*(ptr_A+i - pad_h)+ j - pad_w);
		}
	}


	//Matrix LU(pad_h, pad_w), RU(pad_h, pad_w), LD(pad_h, pad_w), RD(pad_h, pad_w);
	//for (int i = pad_h + 1, k = 0; k < pad_h; i++, k++) {
	//	for (int j = pad_w + 1, z = 0; z < pad_w; j++, z++) {
	//		LU(k, z) = C_tmp(i, j);
	//	}
	//}
	//for (int i = pad_h + 1, k = 0; k < pad_h; i++, k++) {
	//	for (int j = width - 1, z = 0; z < pad_w; j++, z++) {
	//		RU(k, z) = C_tmp(i, j);
	//	}
	//}
	//for (int i = height - 1, k = 0; k < pad_h; i++, k++) {
	//	for (int j = pad_w + 1, z = 0; z < pad_w; j++, z++) {
	//		LD(k, z) = C_tmp(i, j);
	//	}
	//}
	//for (int i = width - 1, k = 0; k < pad_h; i++, k++) {
	//	for (int j = height - 1, z = 0; z < pad_w; j++, z++) {
	//		RD(k, z) = C_tmp(i, j);
	//	}
	//}

	////LU.print(); RU.print(); LD.print(); RD.print();
	//Matrix LU0(pad_h, pad_w), RU0(pad_h, pad_w), LD0(pad_h, pad_w), RD0(pad_h, pad_w);
	//for (int k = 0; k < pad_h; k++) {
	//	for (int z = 0; z < pad_w; z++) {
	//		LU0(pad_h - k - 1, pad_w - z - 1) = LU(k, z);
	//		RU0(pad_h - k - 1, pad_w - z - 1) = RU(k, z);
	//		LD0(pad_h - k - 1, pad_w - z - 1) = LD(k, z);
	//		RD0(pad_h - k - 1, pad_w - z - 1) = RD(k, z);
	//	}
	//}

	//for (int i = 0, k = 0; k < pad_h; i++, k++) {
	//	for (int j = 0, z = 0; z < pad_w; j++, z++) {
	//		C_tmp(i, j) = LU0(k, z);
	//	}
	//}
	//for (int i = 0, k = 0; k < pad_h; i++, k++) {
	//	for (int j = pad_w + width, z = 0; z < pad_w; j++, z++) {
	//		C_tmp(i, j) = RU0(k, z);
	//	}
	//}
	//for (int i = pad_h + height, k = 0; k < pad_h; i++, k++) {
	//	for (int j = 0, z = 0; z < pad_w; j++, z++) {
	//		C_tmp(i, j) = LD0(k, z);
	//	}
	//}
	//for (int i = pad_h + height, k = 0; k < pad_h; i++, k++) {
	//	for (int j = pad_w + width, z = 0; z < pad_w; j++, z++) {
	//		C_tmp(i, j) = RD0(k, z);
	//	}
	//}

	//for (int i = 0; i < 2 * pad_h + height; i++) {
	//	for (int j = 0; j < 2 * pad_w + width; j++) {
	//		if (pad_h <= i&&i < pad_h + height) {
	//			for (int index0 = pad_w + 1, index1 = pad_w + width, k = 0; k < pad_w; index0++, index1++, k++) {
	//				C_tmp(i, index0 - 2 * (k + 1)) = C_tmp(i, index0);
	//				C_tmp(i, index1) = C_tmp(i, index1 - 2 * (k + 1));
	//			}
	//		}
	//		if (pad_w <= j&&j < pad_w + width) {
	//			for (int index0 = pad_h + 1, index1 = pad_h + height, k = 0; k < pad_h; index0++, index1++, k++) {
	//				C_tmp(index0 - 2 * (k + 1), j) = C_tmp(index0, j);
	//				C_tmp(index1, j) = C_tmp(index1 - 2 * (k + 1), j);
	//			}
	//		}
	//	}
	//}


	//return C_tmp;
}
